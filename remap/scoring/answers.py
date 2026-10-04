"""Typed verification of a response's final answer (used as the RL task reward).

Every training row names an answer type.  The response must commit to one answer on
a line ``Answer: ...``; the committed text is normalised for its type (numbers,
class names, sets, image indices, option letters) without looking at the gold answer
and then compared with the accepted answers.  Several ``Answer:`` lines are accepted
only if they all normalise to the same value.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

ANSWER_TYPES = frozenset({"integer", "numeric", "class_set", "class_name", "short_text",
                          "option_text", "public_mcq", "image_index"})

# Appended to every training/evaluation question.
ANSWER_PROTOCOL = (
    "Reason about the visual evidence before giving your final answer. "
    "Put explanations before the final Answer: line. "
    "The Answer: line must contain only the final answer. "
    "For numeric questions, write only the numeric value in the unit requested by the question; "
    "a percent sign is allowed when the question asks for a percentage. "
    "For lists, use comma-separated entries without duplicates. "
    "End with one final line beginning with Answer:."
)

_ANSWER_LINE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*", re.I)
_ANSWER_INLINE = re.compile(r"(?<!\w)(?:final[ \t]+)?answer[ \t]*:[ \t]*", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[。！？]")
_HEDGE = re.compile(r"\b(?:because|therefore|perhaps|maybe|probably|possibly|either|or|and|"
                    r"versus|not|think|believe|is|are|was|were|seems?|appears?|"
                    r"might|could|should)\b")


@dataclass(frozen=True)
class AnswerScore:
    reward: float
    correct: bool
    parsed: bool
    answer: str
    reason: str = ""


def strip_protocol(question: str) -> str:
    suffix = "\n" + ANSWER_PROTOCOL
    return question[:-len(suffix)] if question.endswith(suffix) else question


def final_answer(response: str) -> str:
    """Text after the last ``Answer:`` line (or the last line when there is none)."""
    text = unicodedata.normalize("NFKC", str(response)).strip()
    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        text = text[matches[-1].end():].strip()
    elif "\n" in text:
        text = text.splitlines()[-1].strip()
    if "\n" in text or len(text) > 2048:
        return ""
    return text.strip(" \t`*")


def _text(value) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold().rstrip(".")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip()


def _number(value):
    text = str(value).strip().rstrip(".")
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?%?", text):
        return None
    try:
        number = Decimal(text.removesuffix("%").replace(",", ""))
    except InvalidOperation:
        return None
    return number, text.endswith("%")


def _count(answer, question):
    """A count, optionally followed by the unit the counting question names."""
    value = _number(answer)
    if value is not None:
        return value
    target = re.fullmatch(
        r"how many ([a-z]+(?:[ -][a-z]+)*) instances are there in total across these "
        r"\d+ images\? count objects, including multiple objects in the same image", _text(question))
    match = re.fullmatch(r"([+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?) "
                         r"([a-z]+(?:[ -][a-z]+)*)", _text(answer))
    if target is None or match is None:
        return None
    noun = target.group(1)
    head = noun.rsplit(" ", 1)[-1]
    irregular = {"person": "people", "mouse": "mice", "knife": "knives",
                 "sheep": "sheep", "skis": "skis", "scissors": "scissors"}
    if head in irregular:
        plural = noun[:-len(head)] + irregular[head]
    elif re.search(r"(?:s|x|z|ch|sh)$", head):
        plural = noun + "es"
    elif re.search(r"[^aeiou]y$", head):
        plural = noun[:-1] + "ies"
    else:
        plural = noun + "s"
    units = {noun, plural, noun + " instance", noun + " instances",
             "instance", "instances", "object", "objects"}
    return _number(match.group(1)) if match.group(2) in units else None


def _class_name(answer):
    text = re.sub(r"^(?:a|an|the)\s+", "", _text(answer))
    # A bare noun phrase only: hedged sentences or several candidates do not count.
    if not re.fullmatch(r"[a-z]+(?:[ -][a-z]+)*", text) or _HEDGE.search(text):
        return None
    return text


def _class_set(value):
    text = _text(value).strip("[]{}")
    if text in {"none", "no common objects", "no objects", "empty", "∅"}:
        return frozenset()
    values = [_text(item).strip(" '\"") for item in re.split(r"[,;]|\s+and\s+", text)]
    return frozenset(values) if values and all(values) else None


def validate(reward: dict) -> None:
    kind = reward.get("answer_type")
    if kind not in ANSWER_TYPES:
        raise ValueError(f"unsupported answer type: {kind}")
    if not isinstance(reward.get("gold_answer"), str) or not reward["gold_answer"].strip():
        raise ValueError("reward needs a gold_answer")
    if kind == "public_mcq":
        options = reward.get("public_options")
        if not isinstance(options, dict) or len(options) < 2 or reward.get("correct_letter") not in options:
            raise ValueError("multiple-choice reward needs its visible options")
    if kind == "image_index" and (type(reward.get("answer_index")) is not int or reward["answer_index"] < 0):
        raise ValueError("image-index reward needs a zero-based answer_index")


def _image_labels(options, n_images):
    """Options of the form ``Image k`` -> {letter: k}."""
    if options is None:
        return None
    if not isinstance(options, dict) or not options:
        raise ValueError("image options must be a nonempty mapping")
    labels = {}
    for letter, label in options.items():
        mapped = re.fullmatch(r"Image\s+(\d+)", str(label), flags=re.I)
        if not re.fullmatch(r"[A-Z]", str(letter)) or mapped is None or not 1 <= int(mapped.group(1)) <= n_images:
            raise ValueError("malformed image option")
        labels[letter] = int(mapped.group(1))
    return labels


def _canonical(answer, *, kind, question, n_images, options, image_labels):
    if not answer:
        return None
    if kind in {"integer", "numeric"}:
        value = _count(answer, question) if kind == "integer" else _number(answer)
        if value is None:
            return None
        if kind == "integer" and (value[1] or value[0] != value[0].to_integral_value()):
            return None
        if kind == "numeric" and re.search(r"\b(?:percent|percentage)s?\b", question, flags=re.I):
            return value[0], False      # the question fixes the unit; "97" and "97%" agree
        return value
    if kind == "class_name":
        return _class_name(answer)
    if kind == "class_set":
        return _class_set(answer)
    if kind == "image_index":
        match = re.fullmatch(r"(?:image\s*[-#:]?\s*)?(\d+)\.?", answer, flags=re.I)
        index = int(match.group(1)) if match else None
        if index is None and image_labels is not None:
            letter = re.fullmatch(r"\(?([A-Z])\)?\.?", answer, flags=re.I)
            index = image_labels.get(letter.group(1).upper()) if letter else None
        return index if index is not None and 1 <= index <= n_images else None
    if kind == "public_mcq":
        normalized = _text(answer).strip("() ")
        selected = [letter for letter, text in options.items() if normalized in {_text(letter), _text(text)}]
        return selected[0] if len(selected) == 1 else None
    return _text(answer) or None


def extract_answers(response: str):
    """(answers, number of markers, single clean commitment) for inline ``Answer:`` markers."""
    text = unicodedata.normalize("NFKC", response).strip()
    markers = list(_ANSWER_INLINE.finditer(text))
    if not markers:
        return [text if len(text.splitlines()) == 1 else ""], 0, False
    answers, trailing = [], False
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        segment = text[marker.end():end]
        first_line = segment.splitlines()[0] if segment.splitlines() else ""
        sentence_end = _SENTENCE_END.search(first_line)
        answer_end = sentence_end.start() if sentence_end else len(first_line)
        answers.append(first_line[:answer_end].strip())
        trailing |= bool(segment[answer_end:].strip(" \t\r\n.!?。！？"))
    return answers, len(markers), len(markers) == 1 and not trailing


def public_choice_pair(response: str, options: dict) -> Optional[str]:
    """Letter of a final ``Answer: (B) text`` / ``Answer: B. text`` whose text equals option B."""
    text = unicodedata.normalize("NFKC", response).strip()
    answers, count, clean = extract_answers(text)
    final_line = re.fullmatch(r"(?:final[ \t]+)?answer[ \t]*:[ \t]*(.+)",
                              text.splitlines()[-1].strip(), re.I) if text else None
    if count != 1 or final_line is None:
        return None
    pair = re.fullmatch(r"\(([A-Za-z])\)[ \t]+(.+)", answers[0].strip()) if clean else None
    if pair is None:
        pair = re.fullmatch(r"([A-Za-z])\.[ \t]+(.+)", final_line[1])
    if pair is None:
        return None
    letter = pair[1].upper()
    option = options.get(letter)
    if not isinstance(option, str) or not option.strip() or _normalized(pair[2]) != _normalized(option):
        return None
    return letter


def score_answer(response: str, reward: dict, *, n_images: int, question: str = "",
                 choice_pairs: bool = True) -> AnswerScore:
    """0/1 reward of ``response`` for a row with answer metadata ``reward``.

    ``reward`` holds ``answer_type``, ``gold_answer`` and, depending on the type,
    ``acceptable_answers``, ``public_options`` + ``correct_letter`` or ``answer_index``.
    """
    validate(reward)
    text = unicodedata.normalize("NFKC", str(response)).strip()
    markers = list(_ANSWER_LINE.finditer(text))
    answer = final_answer(text)
    kind = reward["answer_type"]
    options = reward.get("public_options")
    labels = _image_labels(options, n_images) if kind == "image_index" else None
    if labels is not None and (reward.get("correct_letter") not in labels
                               or labels[reward["correct_letter"]] - 1 != reward["answer_index"]):
        raise ValueError("image option and answer_index disagree")
    task_question = strip_protocol(question)
    pair = public_choice_pair(text, options) if choice_pairs and kind == "public_mcq" else None

    def canonical(value):
        if pair is not None and value == answer:
            return pair
        return _canonical(value, kind=kind, question=task_question, n_images=n_images, options=options,
                          image_labels=labels)

    commitments = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        segment = text[marker.end():end].strip()
        commitments.append(segment.splitlines()[0].strip(" \t`*") if segment else "")
    normalized = [canonical(value) for value in commitments] if markers else [canonical(answer)]
    if len({value for value in normalized if value is not None}) > 1:
        return AnswerScore(0., False, False, "", "conflicting_answers")
    if any(value is None for value in normalized):
        return AnswerScore(0., False, False, answer, "unparsed_commitment")
    value = canonical(answer)
    if value is None:
        return AnswerScore(0., False, False, answer, "unparsed_final_answer")
    if kind == "image_index":
        accepted = [reward["answer_index"] + 1]
    elif kind == "public_mcq":
        accepted = [reward["correct_letter"]]
    else:
        accepted = [canonical(item) for item in (reward.get("acceptable_answers") or [reward["gold_answer"]])]
    correct = value in accepted
    return AnswerScore(float(correct), bool(correct), True, answer, "correct" if correct else "incorrect")
