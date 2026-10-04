"""Teacher continuations for collected Stage I states.

For every state the teacher sees the public question, the student's overview images,
the student's own text so far, and pixel proxies of the actual reads (the exact crops
of a Local read; the full images for a Global read).  It never sees the gold answer.

1. *Proposal*: the teacher writes the exact continuation of the student's text after
   the current read: either a complete continuation ending with ``Answer: ...`` or a
   short grounded partial continuation.
2. *Review*: a separate request audits that continuation (grounded in the evidence,
   joins the student prefix literally, uses the current read, no invented facts, and
   the complete/partial judgement).  Only approved continuations are kept.
3. *Answer gate* (offline): a complete continuation's final answer must be correct.

    python -m remap.training.stage1.annotate --states states.jsonl --questions train.jsonl --output annotations/
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...data.teacher import TeacherClient, image_view
from ...scoring.answers import ANSWER_TYPES, score_answer

PROPOSAL_SCHEMA = "fit_exact_suffix_proposal_v1"
REVIEW_SCHEMA = "fit_exact_suffix_review_v1"
SCOPE = "single_current_prefix_and_actual_reads_only"
PROTOCOL_MARKUP = re.compile(r"<\|[^>]*\|>|<mem_[^>]*>|</?s>|<eos>", re.I)
ANSWER_MARKER = re.compile(r"\bAnswer\s*:", re.I)
REVIEW_FLAGS = ("approved", "grounded", "exact_join_compatible", "uses_only_supplied_prefix",
                "current_read_relevant", "no_unseen_facts", "no_latent_identity_claim", "complete_answer",
                "answer_supported", "no_further_read_needed", "partial_continuation_nonterminal")

EVIDENCE_RULE = (
    "This is memory CONTENT training at an already executed read, not action selection or a CALL-necessity audit. "
    "Use only the public question, immutable student prefix, student overview images, and supplied actual prior/current read pixel proxies. "
    "Student claims are unverified hypotheses. Images are evidence proxies, not proof of what compressed latent memory encoded. "
    "Global summarizes whole-image/cross-image evidence and adds no new pixels; Local receives the exact Twig-selected crops. "
    "Never request or invent another ROI. current_read_relevant means the actual current read supports a task-relevant statement in the suffix, "
    "including corroboration or a useful visual relation. It does not mean the read was necessary, indispensable, newly revealing, or better than no read. "
    "Do not reject solely because an overview also shows the same fact; do reject an irrelevant crop, unsupported detail, fabricated perception, "
    "or a suffix that cannot join the immutable prefix. No future student text, gold, teacher prefix, replacement crop or desired route is supplied. "
    "Judge the exact suffix independently. A complete target needs a supported final Answer; a grounded partial may remain unfinished. "
    "Never emit protocol tokens, a new CALL instruction, or claims that this label establishes memory benefit or natural policy quality."
)
JOIN_RULE = (
    "The supplied student prefix is immutable. Return only the exact suffix, preserving every needed "
    "leading space or newline. Evaluate prefix_text + continuation literally, with no implicit separator "
    "and no repair to the prefix. Do not repeat or assume missing prefix text. With an empty prefix, "
    'do not assume an earlier first-image description or begin with a dangling "and" clause. After a '
    'sentence-ending period, do not append ", while ...". The read record is a pixel proxy for compressed '
    "memory, not proof that the student latent encoded every visible detail. Do not say that the student "
    "recognized a fact merely because the reviewer can see it. The suffix is a supervised target only; "
    "it will never replace the student prefix or enter later student sampling. Never emit CALL, memory "
    "END, EOS, role markup, or a new read instruction. "
)
_CURRENT_READ = (
    "A current executed read is supplied separately from prior reads. Ground the suffix in that "
    "actual pixel proxy and earlier visible evidence; reject an irrelevant current read. "
)
_MODE = (
    "Prefer a concise complete answer when the current evidence supports it without another "
    'read. The literal prefix_text + continuation must have exactly one final nonempty line '
    'spelled "Answer: <answer>". Include every necessary newline in continuation; do not put '
    "Answer inline after a sentence. Set complete_answer=true only then. Otherwise provide a "
    "short grounded nonterminal ordinary suffix with complete_answer=false and no Answer "
    "marker. Do not rush an Answer when a needed detail remains unresolved: describe only what "
    "the current evidence supports. Do not invent unresolved issues or more reads to meet coverage. "
)
PROPOSAL_INSTRUCTION = (
    JOIN_RULE + _CURRENT_READ + _MODE +
    'Return {"schema":"' + PROPOSAL_SCHEMA + '","scope":"' + SCOPE + '",'
    '"approved":boolean,"continuation":"exact suffix","complete_answer":boolean,"reason":"brief reason"}.'
)
REVIEW_INSTRUCTION = (
    JOIN_RULE + _CURRENT_READ + _MODE +
    "Independently audit the supplied proposed suffix and concatenated_text literally; "
    "neither proposal nor its mode is evidence. Do not rewrite. exact_join_compatible covers "
    "grammar, discourse, spacing and punctuation; uses_only_supplied_prefix forbids assuming "
    "omitted student text. A self-contained suffix after an empty prefix is allowed. "
    "complete_answer is your own judgment and must match the proposal. A complete target "
    "requires answer_supported and no_further_read_needed, with partial_continuation_nonterminal=false. "
    "A partial target requires partial_continuation_nonterminal=true and both answer_supported "
    "and no_further_read_needed=false. Require current_read_relevant=true. "
    'Return {"schema":"' + REVIEW_SCHEMA + '","scope":"' + SCOPE + '",'
    '"approved":boolean,"grounded":boolean,"exact_join_compatible":boolean,'
    '"uses_only_supplied_prefix":boolean,"current_read_relevant":boolean,'
    '"no_unseen_facts":boolean,"no_latent_identity_claim":boolean,"complete_answer":boolean,'
    '"answer_supported":boolean,"no_further_read_needed":boolean,'
    '"partial_continuation_nonterminal":boolean,"reason":"brief reason"}.'
)


def final_answer_line(text: str):
    """(answer, error): exactly one ``Answer:`` marker, on the last nonempty line, with a value."""
    lines = text.strip().splitlines()
    if not lines:
        return None, "empty"
    if len(re.findall(r"\bAnswer\s*:", text, flags=re.IGNORECASE)) != 1:
        return None, "missing_or_multiple_answer_markers"
    match = re.fullmatch(r"Answer:[ \t]*(\S(?:.*\S)?)", lines[-1])
    return (match.group(1), None) if match else (None, "answer_not_on_final_line")


def check_join(prefix: str, suffix: str, complete: bool) -> None:
    if not isinstance(suffix, str) or not suffix.strip() or PROTOCOL_MARKUP.search(suffix):
        raise ValueError("empty continuation or protocol markup")
    if not prefix.strip() and re.match(r"(?:and\b|[,，;；])", suffix.lstrip(), re.I):
        raise ValueError("dangling continuation after an empty prefix")
    if re.search(r"[.!?。！？]\s*$", prefix) and re.match(r"[,，]", suffix.lstrip()):
        raise ValueError("comma continuation after a completed sentence")
    joined = prefix + suffix
    if complete:
        _, error = final_answer_line(joined)
        if error:
            raise ValueError("complete continuation violates the Answer format: " + error)
    elif ANSWER_MARKER.search(joined):
        raise ValueError("partial continuation contains an Answer marker")


def answer_gate(answer: str, row: dict) -> str:
    """'correct' | 'wrong' | 'unknown' for a final answer of a training question."""
    reward = row["answer"]
    if reward.get("backend") == "typed" and reward.get("answer_type") in ANSWER_TYPES:
        try:
            result = score_answer("Answer: " + answer, reward, n_images=len(row["images"]), question=row["question"])
        except ValueError:
            return "unknown"
        if result.correct:
            return "correct"
        return "wrong" if result.parsed and reward["answer_type"] != "class_name" else "unknown"
    norm = lambda v: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(v))).strip().rstrip(".").casefold()  # noqa: E731
    golds = [reward.get("gold", reward.get("gold_answer", ""))] + list(reward.get("acceptable_answers") or [])
    return "correct" if any(norm(answer) == norm(g) for g in golds if g) else "unknown"


def _views(state: dict, folder: Path):
    """Overview images, then pixel proxies of the previous reads and the current read."""
    parts = []
    for index, path in enumerate(state["images"]):
        parts += [{"type": "text", "text": f"Student overview Image {index + 1} (internal zero-based source index {index})."},
                  image_view(path, budget="main", save_to=folder / f"overview{index}.jpg")]
    reads = [("previous", i, r) for i, r in enumerate(state["previous_reads"])]
    reads.append(("current", len(state["previous_reads"]), state["current_read"]))
    for phase, ordinal, read in reads:
        for index, view in enumerate(read["views"]):
            source = view.get("image_index")
            label = (f"Image {source + 1} (internal zero-based source index {source})" if source is not None
                     else "runtime crop source index unavailable")
            parts += [{"type": "text", "text": f"{phase} actual read {ordinal}, {read['kind']}, {label}; pixel proxy, "
                                               "not a claim that the student decoded every detail."},
                      image_view(view["path"], budget="roi" if read["kind"] == "local" else "main",
                                 save_to=folder / f"{phase}{ordinal}_{read['kind']}{index}.jpg")]
    return parts


def _read_summary(read, index):
    return dict(index=index, kind=read["kind"], image_indices=[v.get("image_index") for v in read["views"]])


def annotate(teacher: TeacherClient, state: dict, row: dict, folder: Path) -> dict:
    """Target record for one collected state (see module docstring)."""
    folder.mkdir(parents=True, exist_ok=True)
    previous = [_read_summary(r, i) for i, r in enumerate(state["previous_reads"])]
    payload = dict(question=state["question"], prefix_text=state["prefix_text"], previous_read_count=len(previous),
                   previous_read_kinds=[r["kind"] for r in previous], previous_reads=previous,
                   current_read=_read_summary(state["current_read"], len(previous)))
    base = dict(state_id=state["state_id"], kind=state["kind"], state=state["state"], status="rejected")
    images = _views(state, folder)
    proposal = teacher.call("Propose an exact ordinary-text suffix from the supplied current evidence. "
                            "Return JSON only. " + EVIDENCE_RULE, images,
                            dict(payload, instruction=PROPOSAL_INSTRUCTION), folder / "proposal.json")
    try:
        if proposal.get("schema") != PROPOSAL_SCHEMA or not isinstance(proposal.get("continuation"), str):
            raise ValueError("invalid proposal schema")
        if not proposal.get("approved"):
            return dict(base, reason="teacher declined: " + str(proposal.get("reason", "")))
        suffix, complete = proposal["continuation"], bool(proposal["complete_answer"])
        check_join(state["prefix_text"], suffix, complete)
    except (ValueError, KeyError) as error:
        return dict(base, reason="invalid proposal: " + str(error))
    review = teacher.call("Independently audit this exact suffix against the supplied evidence. "
                          "Return JSON only. " + EVIDENCE_RULE, images,
                          dict(payload, proposed_continuation=suffix, proposed_complete_answer=complete,
                               concatenated_text=state["prefix_text"] + suffix,
                               target_source="teacher_independent_target", instruction=REVIEW_INSTRUCTION),
                          folder / "review.json")
    if review.get("schema") != REVIEW_SCHEMA or any(type(review.get(k)) is not bool for k in REVIEW_FLAGS):
        return dict(base, reason="invalid review schema")
    approved = (all(review[k] for k in ("approved", "grounded", "exact_join_compatible", "uses_only_supplied_prefix",
                                         "no_unseen_facts", "no_latent_identity_claim", "current_read_relevant"))
                and review["complete_answer"] == complete)
    if complete:
        approved = approved and review["answer_supported"] and review["no_further_read_needed"] \
            and not review["partial_continuation_nonterminal"]
    else:
        approved = approved and review["partial_continuation_nonterminal"] and not review["answer_supported"] \
            and not review["no_further_read_needed"]
    if not approved:
        return dict(base, reason="review rejected: " + str(review.get("reason", "")))
    gate = "not_applicable"
    if complete:
        answer, _ = final_answer_line(state["prefix_text"] + suffix)
        gate = answer_gate(answer, row)
        if gate != "correct":
            return dict(base, status="answer_" + gate, suffix_text=suffix)
    return dict(base, status="approved", suffix_text=suffix, endpoint="final_answer" if complete else "partial",
                answer_gate=gate, reason=review.get("reason", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="states.jsonl written by stage1.collect")
    parser.add_argument("--questions", required=True, help="the training questions (with answers)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--rps", type=float, default=9.5)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in map(json.loads, Path(args.questions).read_text().splitlines()) if r}
    states = [json.loads(line) for line in Path(args.states).read_text().splitlines() if line.strip()]
    teacher = TeacherClient(rps=args.rps)

    def work(state):
        path = output / state["state_id"] / "target.json"
        if path.exists():
            return json.loads(path.read_text())
        try:
            target = annotate(teacher, state, rows[state["row_id"]], output / state["state_id"])
        except Exception as error:  # transport failures are recorded, not silently dropped
            target = dict(state_id=state["state_id"], status="error", reason=f"{type(error).__name__}: {error}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(target, ensure_ascii=False, indent=1))
        return target

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        targets = list(pool.map(work, states))
    with (output / "manifest.jsonl").open("w") as stream:
        for target in targets:
            if target["status"] == "approved":
                stream.write(json.dumps(dict(state=target["state"], target=str(output / target["state_id"] / "target.json"),
                                             kind=target["kind"])) + "\n")
    counts = {}
    for target in targets:
        counts[target["status"]] = counts.get(target["status"], 0) + 1
    print(json.dumps(dict(states=len(states), status=counts, usage=dict(teacher.usage))))


if __name__ == "__main__":
    main()
