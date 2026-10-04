"""Branch selection, advantages and the policy loss for Stage II (memory access)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


# ------------------------------------------------------------------ branch selection (CRD)
def route_probability(p_global: float, p_local: float, action: str) -> float:
    """Probability of the realised route: defer (no request), global or local."""
    if action == "defer":
        value = 1.0 - p_global - p_local
    elif action == "global":
        value = p_global
    elif action == "local":
        value = p_local
    else:
        raise ValueError(action)
    return min(1.0, max(0.0, value))


def conditional_route_divergence(p_global: float, p_local: float, action: str, siblings: int) -> float:
    """u = 1 - p(a)^m: probability that at least one of m siblings takes a different route."""
    return 1.0 - route_probability(p_global, p_local, action) ** int(siblings)


@dataclass(frozen=True)
class BranchPoint:
    root: int
    step: int
    action: str          # realised route at the branch state: defer / global / local
    score: float


def select_branch_points(roots, *, siblings: int, min_step: int, max_step_frac: float,
                         max_new: int) -> List[BranchPoint]:
    """One branch per root, at the state that maximises CRD (ties: earlier step)."""
    max_step = int(max_new * max_step_frac)
    points = []
    for index, root in enumerate(roots):
        limit = min(max_step, root.answer_commit_step - 1 if root.answer_commit_step is not None else max_step)
        candidates = []
        for state in root.route_states:
            if min_step <= state.step <= limit:
                score = conditional_route_divergence(state.p_global, state.p_local, state.action, siblings)
                candidates.append(BranchPoint(index, state.step, state.action, score))
        if candidates:
            points.append(max(candidates, key=lambda p: (p.score, -p.step)))
    return points


# ------------------------------------------------------------------ advantages
def group_advantage(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """GRPO advantage over one prompt group (unbiased std); all-equal rewards give zeros."""
    values = rewards.detach().float().cpu()
    if torch.equal(values, values[0].expand_as(values)):
        return torch.zeros_like(values).to(rewards.device)
    if len(values) == 1:
        mean, std = torch.tensor(0.0), torch.tensor(1.0)
    else:
        mean, std = values.mean(), values.std()
    advantage = torch.empty_like(values)
    for i in range(len(values)):
        advantage[i] = (values[i] - mean) / (std + epsilon)
    return advantage.to(rewards.device)


def memory_call_advantages(trajectories, branches: Sequence[BranchPoint], rewards: torch.Tensor, kind_of, *,
                           no_call_reward: Optional[float], epsilon: float = 1e-6) -> torch.Tensor:
    """Credit for requesting memory at the branch state.

    For each branched root, trajectories whose route at the branch step is a memory
    request get (R_i - B) / (std + eps), where B is the mean reward of the trajectories
    that deferred at the same state (or, if none did, the reward of a matched rollout
    with memory disabled) and std is taken over the root's rewards together with B.
    """
    by_root = {p.root: p for p in branches}
    indices = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        indices[trajectory.root].append(index)
    result = torch.zeros_like(rewards, dtype=torch.float32)
    for root, members in sorted(indices.items()):
        point = by_root.get(root)
        if point is None:
            continue
        actions = []
        for index in members:
            ids = trajectories[index].token_ids
            actions.append((kind_of(int(ids[point.step])) or "defer") if point.step < len(ids) else "defer")
        defer = [index for index, action in zip(members, actions) if action == "defer"]
        if defer:
            control = rewards[torch.as_tensor(defer, device=rewards.device)].mean()
        elif no_call_reward is not None:
            control = rewards.new_tensor(float(no_call_reward))
        else:
            continue
        joined = torch.cat([rewards[torch.as_tensor(members, device=rewards.device)], control.view(1)])
        scale = joined.std(unbiased=True) if joined.numel() > 1 else joined.new_tensor(0.0)
        if not torch.isfinite(scale) or float(scale) <= epsilon:
            scale = joined.new_tensor(1.0)
        for index, action in zip(members, actions):
            if action in ("global", "local"):
                result[index] = (rewards[index] - control) / (scale + epsilon)
    return result


def memory_credit_start(trajectory, point: Optional[BranchPoint], kind_of) -> Optional[int]:
    """Response index of the request token at the branch step (None if it is not a request)."""
    if point is None or not 0 <= point.step < len(trajectory.token_ids):
        return None
    return point.step if kind_of(int(trajectory.token_ids[point.step])) is not None else None


# ------------------------------------------------------------------ loss
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def dual_clip_policy_loss(old_logp, logp, advantages, mask, *, clip: float = 0.2,
                          clip_c: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token dual-clip PPO loss; also returns clip fraction and approximate KL."""
    log_ratio = logp - old_logp
    ratio = torch.exp(log_ratio)
    approx_kl = masked_mean(-log_ratio, mask)
    loss1 = -advantages * ratio
    loss2 = -advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
    clipped = torch.maximum(loss1, loss2)
    clip_fraction = masked_mean(torch.gt(loss2, loss1).float(), mask)
    lower = torch.min(-advantages * clip_c, clipped)
    return torch.where(advantages < 0, lower, clipped), clip_fraction, approx_kl
