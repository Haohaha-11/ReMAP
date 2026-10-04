"""Stage II reward: task score plus a bounded, decaying bonus for memory reads.

    reward = task + task * scale * sum(bonus[:calls]) - cost * max(0, calls - 1)
    scale  = 1 - (1 - final_fraction) * progress

``task`` is in [0, 1].  Only a correct answer earns the bonus, only the first
``len(bonus)`` reads count, and the bonus shrinks linearly to ``final_fraction`` of
its initial value over training.  The first read is free; later reads cost a little.

Task scores per row (``row["answer"]["backend"]``):

* ``typed``: :func:`remap.scoring.answers.score_answer` on the typed answer metadata;
* ``mimic``: the MIMIC metric on the ``Answer:`` payload;
* ``judge``: an LLM judge (MM-Vet / MathVista free-form), cached per final segment.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from ...scoring.answers import score_answer
from ...scoring.benchmarks import final_segment, mimic_score, typed_final_payload


@dataclass
class RewardSettings:
    correct_call_bonus: Tuple[float, ...] = (0.12, 0.06, 0.03)
    bonus_final_fraction: float = 0.25
    additional_call_cost: float = 0.002
    max_calls: int = 5


def shaped_reward(task: float, calls: int, progress: float, settings: RewardSettings):
    """(reward, bonus, cost, scale)."""
    assert math.isfinite(task) and 0 <= task <= 1
    assert isinstance(calls, int) and 0 <= calls <= settings.max_calls
    assert 0 <= progress <= 1
    scale = 1 - (1 - settings.bonus_final_fraction) * progress
    bonus = task * scale * sum(settings.correct_call_bonus[:calls])
    cost = settings.additional_call_cost * max(0, calls - 1)
    return task + bonus - cost, bonus, cost, scale


class TaskScorer:
    """Task score of a response for one training row."""

    def __init__(self, row: dict, judge=None, log_path: Optional[Path] = None):
        self.row, self.answer = row, row["answer"]
        self.judge, self.log_path = judge, Path(log_path) if log_path else None
        self.cache = {}

    def __call__(self, text: str):
        backend = self.answer["backend"]
        if backend == "typed":
            result = score_answer(text, self.answer, n_images=len(self.row["images"]), question=self.row["question"])
            return float(result.reward), dict(parsed=result.parsed, correct=result.correct)
        if backend == "mimic":
            payload, rule = typed_final_payload(text)
            value, correct = mimic_score(self.row["benchmark"], self.answer["gold"], payload or "")
            return float(value), dict(parsed=payload is not None, correct=bool(correct), payload_rule=rule)
        if backend != "judge":
            raise ValueError(f"unknown reward backend: {backend}")
        if self.judge is None:
            raise RuntimeError("rows with the judge backend need an LLM judge")
        segment = final_segment(text)
        if segment not in self.cache:
            result = self.judge.score(self.row["benchmark"], self.answer["question"], self.answer["gold"], text)
            self.cache[segment] = float(result["score"])
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as stream:
                    stream.write(json.dumps(dict(row=self.row["id"], **result), ensure_ascii=False) + "\n")
        score = self.cache[segment]
        return score, dict(parsed=True, correct=score == 1.)


@dataclass
class RewardFunction:
    """Assigns ``trajectory.reward``; keeps one diagnostic record per scored trajectory."""
    scorer: TaskScorer
    tokenizer: object
    token_ids: object
    progress: float
    settings: RewardSettings
    records: list = field(default_factory=list)

    def __call__(self, trajectory) -> float:
        text = self.tokenizer.decode(trajectory.token_ids, skip_special_tokens=True)
        task, detail = self.scorer(text)
        calls = [kind for kind in map(self.token_ids.kind, trajectory.token_ids) if kind is not None]
        value, bonus, cost, scale = shaped_reward(task, len(calls), self.progress, self.settings)
        trajectory.reward = value
        self.records.append(dict(root=trajectory.root, variant=trajectory.variant, task=task, calls=calls,
                                 bonus=bonus, cost=cost, scale=scale, reward=value, **detail))
        return value


def reward_function(model, row: dict, progress: float, settings: RewardSettings, judge=None,
                    log_path: Optional[Path] = None) -> RewardFunction:
    return RewardFunction(TaskScorer(row, judge, log_path), model.processor.tokenizer, model.token_ids,
                          progress, settings)
