"""Stage II: learn when and which memory to request.

Trainable parameters: the policy LoRA (top four decoder layers) and the input/output
rows of the two request tokens.  Every memory module, the END rows and the base model
stay frozen; the latent blocks of a rollout are reused when it is scored.

One update per question:

* rewards for all trajectories of the tree; a GRPO advantage over the group applies
  to every sampled token of a trajectory (children include the copied prefix);
* a memory-call advantage (request vs. defer at the same branch state) applies to
  the tokens from the branch's request onward and only updates the request rows;
* dual-clip PPO, gradients averaged across ranks, norm clipped at ``grad_clip``.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.distributed as dist

from ..common import seed_all  # noqa: F401  (re-exported for the entry points)
from ..offload import offload_activations
from .objectives import (dual_clip_policy_loss, group_advantage, memory_call_advantages, memory_credit_start)
from .rollout import TreeConfig, tree_rollout
from .scoring import build_layout, sequence_logprobs

ADVANTAGE_TOL = 1e-8
GRAD_TOL = 1e-6


@dataclass
class RLConfig:
    learning_rate: float = 2e-6
    clip: float = 0.2
    clip_c: float = 3.0
    grad_clip: float = 1.0
    ppo_epochs: int = 1
    gradient_bucket_bytes: int = 25 * 1024 * 1024
    offload_min_tokens: int = 7680      # keep autograd activations on the CPU for longer sequences


def policy_parameters(model):
    """(name, parameter) of the trainable policy: LoRA tensors, then the two request rows."""
    named = [(n, p) for n, p in model.named_parameters() if ".lora_A.policy." in n or ".lora_B.policy." in n]
    named += [("memory_rows.embed_request", model.memory_rows.embed_request),
              ("memory_rows.head_request", model.memory_rows.head_request)]
    return named


def build_optimizer(model, cfg: RLConfig):
    model.requires_grad_(False)
    named = policy_parameters(model)
    for _, parameter in named:
        parameter.requires_grad_(True)
    lora = [p for n, p in named if ".policy." in n]
    rows = [p for n, p in named if n.startswith("memory_rows.")]
    optimizer = torch.optim.AdamW([{"params": lora, "weight_decay": 0.0}, {"params": rows, "weight_decay": 0.0}],
                                  lr=cfg.learning_rate, betas=(0.9, 0.95))
    return optimizer, lora, rows


@torch.no_grad()
def average_gradients_bucketed(params: List[torch.nn.Parameter], bucket_bytes: int) -> None:
    """All-reduce mean of gradients in fixed-size buckets (missing gradients count as zero)."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    world = dist.get_world_size()
    groups = {}
    for param in params:
        groups.setdefault((param.device, param.dtype), []).append(param)
    for (device, dtype), group in groups.items():
        flats = []
        for param in group:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            flats.append(param.grad.reshape(-1))
        numel = sum(flat.numel() for flat in flats)
        if not numel:
            continue
        capacity = min(bucket_bytes // group[0].element_size(), numel)
        buffer = torch.empty(capacity, device=device, dtype=dtype)
        segments, used = [], 0

        def flush():
            nonlocal used
            if not used:
                return
            active = buffer[:used]
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
            active.div_(world)
            offset = 0
            for flat, start, count in segments:
                flat[start:start + count].copy_(active[offset:offset + count])
                offset += count
            segments.clear()
            used = 0

        for flat in flats:
            start = 0
            while start < flat.numel():
                count = min(capacity - used, flat.numel() - start)
                buffer[used:used + count].copy_(flat[start:start + count])
                segments.append((flat, start, count))
                used += count
                start += count
                if used == capacity:
                    flush()
        flush()


def _any_rank(flag: bool, device) -> bool:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return flag
    value = torch.tensor(int(flag), device=device, dtype=torch.int32)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def rl_content(question: str, image_count: int) -> list:
    """Stage II prompt: all images first, then the question text."""
    return [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": question}]


def encode_prompt(model, images, question: str):
    prompt = model.processor.apply_chat_template([{"role": "user", "content": rl_content(question, len(images))}],
                                                 tokenize=False, add_generation_prompt=True)
    encoded = model.processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
    return encoded["input_ids"][0].tolist(), encoded["pixel_values"], encoded["image_grid_thw"]


def rl_step(model, optimizer, lora, rows, prompt, question: str, reward_fn: Callable,
            tree_cfg: TreeConfig, cfg: RLConfig) -> dict:
    """Rollout, score and update on one question.  ``prompt`` is a prefilled PromptCache."""
    device = model.device
    tokenizer = model.processor.tokenizer
    kind_of = model.token_ids.kind
    eos = int(tokenizer.eos_token_id)
    prompt_ids, pixel_values, grid = encode_prompt(model, prompt.state.images, question)
    if prompt_ids != prompt.state.input_ids[0].tolist() or not torch.equal(grid, prompt.state.grid):
        raise RuntimeError("scoring prompt differs from the rollout prompt")
    tree = tree_rollout(model, prompt, tree_cfg, reward_fn)
    no_call_reward = float(tree.control.reward) if tree.control is not None else None

    specs = []
    for trajectory in tree.trajectories:
        generated = list(trajectory.token_ids) + ([eos] if trajectory.stop_reason == "eos" else [])
        layout = build_layout(prompt_ids, generated, model.token_ids)
        reader = next((e.reader for e in reversed(trajectory.events) if e.kind == "local"), None)
        with torch.no_grad():
            old = sequence_logprobs(model, layout, trajectory.blocks, pixel_values, grid, reader_memory=reader)
        specs.append((layout, trajectory.blocks, reader, old.detach()))
    rewards = torch.tensor([float(t.reward) for t in tree.trajectories], dtype=torch.float32, device=device)
    advantages = group_advantage(rewards)
    memory_advantages = memory_call_advantages(tree.trajectories, tree.branches, rewards.detach().float().cpu(),
                                               kind_of, no_call_reward=no_call_reward).to(device)
    by_root = {p.root: p for p in tree.branches}
    starts = [memory_credit_start(t, by_root.get(t.root), kind_of) for t in tree.trajectories]
    total_tokens = sum(int(old.numel()) for *_, old in specs)
    signal = bool(total_tokens > 0 and (advantages.abs().max() > ADVANTAGE_TOL
                                        or memory_advantages.abs().max() > ADVANTAGE_TOL))
    stats = dict(reward_mean=float(rewards.mean()), trajectories=len(tree.trajectories),
                 branches=len(tree.branches), tokens=total_tokens, loss=0.0, stepped=False)
    if not _any_rank(signal, device):
        optimizer.zero_grad(set_to_none=True)
        return stats
    for _ in range(cfg.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        credit = {id(p): torch.zeros_like(p) for p in rows}
        credit_tokens = sum(max(0, int(old.numel()) - int(start)) for start, (*_, old) in zip(starts, specs)
                            if start is not None)
        epoch_loss = 0.0
        if signal:
            for index, (layout, blocks, reader, old) in enumerate(specs):
                if old.numel() == 0:
                    continue
                offload = len(layout.ids) + sum(int(b.shape[1]) for b in blocks) >= cfg.offload_min_tokens
                with offload_activations(model) if offload else nullcontext():
                    logp = sequence_logprobs(model, layout, blocks, pixel_values, grid, reader_memory=reader)
                mask = torch.ones_like(logp)
                losses, _, _ = dual_clip_policy_loss(old, logp, torch.full_like(logp, float(advantages[index])),
                                                     mask, clip=cfg.clip, clip_c=cfg.clip_c)
                loss = (losses * mask).sum() / float(total_tokens)
                if (starts[index] is not None and abs(float(memory_advantages[index])) > ADVANTAGE_TOL
                        and credit_tokens > 0):
                    credit_mask = torch.zeros_like(logp)
                    credit_mask[int(starts[index]):] = 1
                    credit_losses, _, _ = dual_clip_policy_loss(
                        old, logp, torch.full_like(logp, float(memory_advantages[index])), credit_mask,
                        clip=cfg.clip, clip_c=cfg.clip_c)
                    credit_loss = (credit_losses * credit_mask).sum() / float(credit_tokens)
                    grads = torch.autograd.grad(credit_loss, rows, retain_graph=True, allow_unused=True)
                    for parameter, gradient in zip(rows, grads):
                        if gradient is not None:
                            credit[id(parameter)].add_(gradient.detach())
                epoch_loss += float(loss.detach())
                loss.backward()
        for parameter in rows:
            gradient = credit[id(parameter)]
            parameter.grad = gradient if torch.count_nonzero(gradient).item() else None
        params = lora + rows
        average_gradients_bucketed(params, cfg.gradient_bucket_bytes)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("non-finite policy gradient")
        if float(norm) > GRAD_TOL:
            optimizer.step()
            stats["stepped"] = True
        stats["loss"] = epoch_loss
    return stats
