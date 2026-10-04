"""Benchmark metrics: answer extraction and aggregation for every evaluated benchmark.

The multiple-choice parsers follow the official evaluation code of each benchmark
(VLMEvalKit's ``can_infer`` for most, the BLINK and MMSI release parsers for those
two).  MIMIC tasks use the released MIMIC metrics on the raw output.  MathVista
free-form answers and MM-Vet need an LLM judge (:mod:`remap.scoring.judge`).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Optional, Sequence

from .answers import ANSWER_PROTOCOL

BUILTIN = frozenset({"vstar", "cvbench_2d", "cvbench_3d", "blink", "muirbench"})
MIMIC = frozenset({"mimic_common", "mimic_counting", "mimic_oddone", "mimic_listing"})
HELDOUT = frozenset({"mathvista", "mmvet"})
HR = frozenset({"hrbench4k", "hrbench8k"})
TASKS = BUILTIN | MIMIC | HELDOUT | HR | {"mmsi", "realworldqa"}

_CHOICE_FINAL = "\nFor this multiple-choice question, put only the correct option letter on the final Answer: line."
# Benchmark prompts end with "answer directly" instructions; they are replaced by the
# equivalent instruction for the final Answer: line.
DIRECT_ANSWER_TRAILERS = (
    ("\nHint: Please provide the correct option letter, such as A, B, C, D, directly.\nAnswer:", _CHOICE_FINAL),
    ("\nAnswer with the option's letter from the given choices directly. Enclose the option's letter within ``.",
     "\nOn the final Answer: line, provide only the correct option letter enclosed in backticks."),
    (" Please provide the correct class name directly.\nAnswer:",
     "\nOn the final Answer: line, provide only the correct class name."),
    (" Provide the precise class name directly.\nAnswer:",
     "\nOn the final Answer: line, provide only the precise class name."),
    ("\nPlease provide the correct number directly.\nAnswer:",
     "\nOn the final Answer: line, provide only the correct number."),
    (" Please provide the list of category names directly.\nAnswer:",
     "\nOn the final Answer: line, provide only the requested list of category names."),
    ("\nAnswer the option letter directly.", _CHOICE_FINAL),
)


def project_question(question: str) -> str:
    """Benchmark prompt -> ReMAP prompt (reason first, then one final ``Answer:`` line)."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("benchmark question must be nonempty text")
    suffix = "\n" + ANSWER_PROTOCOL
    value = question[:-len(suffix)] if question.endswith(suffix) else question
    for trailer, replacement in DIRECT_ANSWER_TRAILERS:
        if value.endswith(trailer):
            value = value[:-len(trailer)] + replacement
            break
    return value + suffix


def generation_budget(task: str) -> int:
    return 2048 if task in HR | {"mathvista"} else 768


# ---------------------------------------------------------------- choice parsers
def can_infer(answer, choices: dict):
    """VLMEvalKit ``can_infer``: an option letter, or False."""
    text = str(answer)
    if "Failed to obtain answer via API" not in text:
        if any(error in text for error in ("Sorry, I can't help with images of people yet.",
                                           "I can't process this file.",
                                           "I'm sorry, but without the image provided",
                                           "Cannot determine the answer")):
            return "Z"
        normalized = text
        for char in ".()[],:;!*#{}":
            normalized = normalized.replace(char, " ")
        pieces = [piece.strip() for piece in normalized.split()]
        present = [choice for choice in choices if choice in pieces]
        if len(present) == 1 and pieces.index(present[0]) > len(pieces) - 5:
            return present[0]
        match = re.search(r"(?i)(?:correct\s+)?answer\s+is\s+\**([A-H])\**", text)
        if match and match.group(1).upper() in choices:
            return match.group(1).upper()
    lowered = text.lower()
    if len(lowered) > 2 * sum(len(str(value)) for value in choices.values()):
        return False
    found = [key for key, value in choices.items() if str(value).lower() in lowered]
    return found[0] if len(found) == 1 else False


def infer_blink_answer(answer, choices: dict):
    """BLINK release parser: an option letter, or False."""
    text, letters = str(answer).strip(), set(choices)
    if text in letters:
        return text
    if text in {f"({letter})" for letter in letters}:
        return text[1]
    markers = {f"({letter})": letter for letter in letters}
    all_markers = {letter for marker, letter in markers.items() if marker in set(text.split())}
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    last = paragraphs[-1] if paragraphs else text
    last_markers = {letter for marker, letter in markers.items() if marker in set(last.split())}
    if len(last_markers) == 1:
        return next(iter(last_markers))
    if len(all_markers) == 1:
        return next(iter(all_markers))
    blocks = [last, "\n\n".join(paragraphs[-2:])] if len(paragraphs) > 1 else [last]
    for block in blocks:
        found = {letter.upper() for letter in re.findall(r"\(([A-H])\)", block, re.I) if letter.upper() in letters}
        if len(found) == 1:
            return next(iter(found))
    patterns = (r"(?is)(?:correct\s+)?answer\s+(?:is|:)\s*[*_\s]*(?:\()?([A-H])(?:\))?",
                r"(?is)(?:therefore|thus|so),?[^\n]{0,120}?(?:answer|choice|point)\s+(?:is|:)\s*[*_\s]*(?:\()?([A-H])(?:\))?")
    cued = [(match.start(), match.group(1).upper()) for pattern in patterns for match in re.finditer(pattern, text)
            if match.group(1).upper() in letters]
    if cued:
        position, letter = max(cued)
        found = {item.upper() for item in re.findall(r"\(([A-H])\)", text[position:], re.I) if item.upper() in letters}
        return letter if len(found) <= 1 else False
    found = [letter for letter, value in choices.items() if str(value).strip() and str(value).lower() in last.lower()]
    return found[0] if len(found) == 1 else False


def mmsi_choice(prediction: str) -> str:
    """MMSI-Bench release parser."""
    text = str(prediction)
    for pattern in (r"``([^`]*)``", r"`([^`]*)`", r"\{([^}]*)\}"):
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
    match = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", text)
    return match.group().upper() if match else ""


# ---------------------------------------------------------------- MIMIC
_SMALL = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                  "fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))


def count_value(text: str) -> Optional[int]:
    """MIMIC counting parser (digits or English number words)."""
    response = str(text)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    try:
        return int(response)
    except ValueError:
        pass
    words = [word for word in re.findall(r"[a-z]+", response.lower().replace("-", " "))
             if word in _SMALL or word in _TENS or word in {"hundred", "thousand", "and"}]
    if words and any(word != "and" for word in words):
        total = current = 0
        for word in words:
            if word in _SMALL:
                current += _SMALL[word]
            elif word in _TENS:
                current += _TENS[word]
            elif word == "hundred":
                current = max(1, current) * 100
            elif word == "thousand":
                total += max(1, current) * 1000
                current = 0
        return total + current
    matches = re.findall(r"\d+", response)
    return int(matches[0]) if matches else None


def _listing_items(text: str) -> list:
    return [item.lower() for item in str(text).split(", ")]


def listing_micro_f1(rows: Sequence[dict]):
    recalled = generated = gold_total = 0
    for row in rows:
        pred, gold = _listing_items(row["prediction"]), _listing_items(row["gold"])
        generated += len(pred)
        gold_total += len(gold)
        recalled += sum(any(item == candidate for candidate in pred) for item in gold)
    precision = recalled / generated if generated else 0.
    recall = recalled / gold_total if gold_total else 0.
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return score, {"recalled": recalled, "generated_items": generated, "gold_items": gold_total}


def mimic_score(task: str, gold: str, prediction: str):
    """(score, correct) of one MIMIC answer."""
    if task == "mimic_counting":
        try:
            target = int(gold.strip())
        except ValueError:
            return 0., False
        correct = count_value(prediction) == target
    elif task == "mimic_listing":
        score, _ = listing_micro_f1([{"prediction": prediction, "gold": gold}])
        return score, _listing_items(prediction) == _listing_items(gold)
    else:
        correct = gold.lower().strip() in prediction.lower().strip()
    return float(correct), correct


def typed_final_payload(prediction: str):
    """(payload, rule): the text of the ``Answer:`` line(s); None if missing or inconsistent."""
    text = prediction.strip()
    markers = list(re.finditer(r"(?:^|\n)[ \t]*(?:final[ \t]+)?answer[ \t]*:[ \t]*", text, re.I))
    if not markers:
        if text and "\n" not in text and "\r" not in text:
            return text.strip(" \t`*"), "single_line_answer_only"
        return None, "missing_answer_marker"
    payloads = []
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        payload = text[marker.end():stop].strip().strip("`*")
        if not payload or "\n" in payload or "\r" in payload:
            return None, "empty_or_multiline_payload"
        payloads.append(payload)
    if len({" ".join(value.casefold().split()) for value in payloads}) != 1:
        return None, "conflicting_payloads"
    return payloads[-1], "answer_payload"


def final_segment(text, limit: int = 400) -> str:
    """The part of a response an LLM judge sees: from the last answer cue, at most ``limit`` chars."""
    value = str(text)
    lowered = value.lower()
    position = max(lowered.rfind(marker) for marker in ("answer:", "final answer", "therefore"))
    return value[position:position + limit] if position >= 0 else value[-limit:]


# ---------------------------------------------------------------- scoring
def score_prediction(task: str, raw: str, gold: str, options: Optional[dict] = None, qtype: str = "mcq") -> dict:
    """Score one successful generation.  ``judge_required`` marks rows left to the LLM judge."""
    options = options or {}
    gold = str(gold).strip()
    record = dict(score=0., correct=False, pred_letter="", judge_required=False, prediction=raw, gold=gold)
    if task in HELDOUT:
        if task == "mathvista" and qtype == "mcq":
            inferred = can_infer(final_segment(raw), options)
            inferred = inferred if inferred in options else ""
            record.update(pred_letter=inferred, correct=inferred == gold, score=float(inferred == gold))
        else:
            record.update(score=None, correct=None, judge_required=True, judge_prediction=final_segment(raw))
        return record
    if task in MIMIC:
        # Released MIMIC metrics on the raw output (corpus micro-F1 for listing); the score on the
        # final Answer line alone is kept as a diagnostic.
        score, correct = mimic_score(task, gold, raw)
        payload, rule = typed_final_payload(raw)
        record.update(score=score, correct=correct, payload_rule=rule,
                      payload_score=mimic_score(task, gold, payload)[0] if payload is not None else 0.0)
        return record
    inferred = (infer_blink_answer(raw, options) if task == "blink" else
                mmsi_choice(raw) if task == "mmsi" else can_infer(raw, options))
    inferred = str(inferred) if inferred and inferred != "Z" else ""
    if task in HR and inferred not in {"A", "B", "C", "D"}:
        inferred = ""
    record.update(pred_letter=inferred, correct=inferred == gold.upper(), score=float(inferred == gold.upper()))
    return record


def aggregate(task: str, records: Sequence[dict], metadata: Sequence[dict]):
    """(score, rule, details) over a complete benchmark."""
    if task == "mimic_listing":
        score, details = listing_micro_f1(records)
        return score, "corpus_micro_f1", details
    if task in HR | {"cvbench_2d", "blink"}:
        field = "cycle_category" if task in HR else "source" if task == "cvbench_2d" else "sub_task"
        groups = defaultdict(list)
        for record, meta in zip(records, metadata):
            groups[meta[field]].append(record["score"])
        means = {str(key): sum(values) / len(values) for key, values in groups.items()}
        rule = ("macro_mean_over_cycle_categories" if task in HR else
                "mean_source_accuracy" if task == "cvbench_2d" else "mean_subtask_accuracy")
        return sum(means.values()) / len(means), rule, {"group_scores": means}
    score = sum(row["score"] for row in records) / len(records)
    if not math.isfinite(score):
        raise FloatingPointError(f"non-finite {task} score")
    return score, "micro_accuracy", {}
