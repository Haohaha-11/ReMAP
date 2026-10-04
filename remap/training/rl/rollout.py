"""Branched rollouts for Stage II.

For one question:

1. A matched control rollout with memory requests disabled (random state restored
   afterwards, so it does not shift the policy samples).
2. ``roots`` complete rollouts sampled at temperature 1, recording the routing
   probabilities p(<mem_g>), p(<mem_l>) at every decision.
3. For each root, one branch state chosen by conditional route divergence
   (:func:`select_branch_points`); the root's prefix up to that state is replayed
   (memory re-inserted) and ``siblings`` continuations are sampled from it.

Every trajectory is scored by ``reward_fn`` as soon as it finishes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import torch

from ...inference.session import DecodeConfig, Session, Trajectory
from .objectives import BranchPoint, select_branch_points


@dataclass
class TreeConfig:
    roots: int = 4
    siblings: int = 2
    max_new_tokens: int = 768
    max_calls: int = 5
    answer_grace: int = 64
    temperature: float = 1.0
    min_branch_step: int = 1
    max_branch_frac: float = 0.95

    def decode_config(self, **overrides) -> DecodeConfig:
        config = DecodeConfig(max_new_tokens=self.max_new_tokens, max_calls=self.max_calls,
                              answer_grace=self.answer_grace, temperature=self.temperature)
        return replace(config, **overrides)


@dataclass
class Tree:
    control: Optional[Trajectory]
    roots: List[Trajectory]
    branches: List[BranchPoint]
    trajectories: List[Trajectory]


@torch.no_grad()
def matched_control(model, prompt, cfg: TreeConfig, reward_fn: Callable) -> Trajectory:
    device = model.device
    python_state = random.getstate()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        trajectory = Session(model, prompt).decode(cfg.decode_config(allow_memory=False))
        reward_fn(trajectory)
        return trajectory
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


@torch.no_grad()
def tree_rollout(model, prompt, cfg: TreeConfig, reward_fn: Callable, *, record_logprobs: bool = False,
                 control: bool = True) -> Tree:
    control_trajectory = matched_control(model, prompt, cfg, reward_fn) if control else None
    roots = []
    for index in range(cfg.roots):
        root = Session(model, prompt).decode(cfg.decode_config(record_routes=True, record_logprobs=record_logprobs))
        root.root = index
        reward_fn(root)
        roots.append(root)
    branches = select_branch_points(roots, siblings=cfg.siblings, min_step=cfg.min_branch_step,
                                    max_step_frac=cfg.max_branch_frac, max_new=cfg.max_new_tokens)
    trajectories = list(roots)
    for point in branches:
        prefix = roots[point.root].token_ids[:point.step]
        session = Session(model, prompt)
        session.replay(prefix)
        snapshot = session.checkpoint()
        for variant in range(cfg.siblings):
            session.restore(snapshot)
            child = session.decode(cfg.decode_config(record_logprobs=record_logprobs),
                                   max_steps=max(1, cfg.max_new_tokens - point.step))
            trajectory = Trajectory(
                token_ids=list(prefix) + child.token_ids, events=list(snapshot["events"]) + child.events,
                stop_reason=child.stop_reason,
                answer_commit_step=(point.step + child.answer_commit_step
                                    if child.answer_commit_step is not None else None),
                logprobs=child.logprobs, root=point.root, variant=variant + 1, branch_step=point.step)
            reward_fn(trajectory)
            trajectories.append(trajectory)
    return Tree(control_trajectory, roots, branches, trajectories)
