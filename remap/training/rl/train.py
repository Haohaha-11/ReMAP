"""Stage II training entry point.

    torchrun --nproc_per_node 8 -m remap.training.rl.train --config configs/stage2_rl.yaml

One process per GPU; every process trains on its own question at each step (one
"group") and gradients are averaged across processes.  Rows are JSON lines:

    {"id": ..., "benchmark": ..., "images": [...], "question": ..., "max_new_tokens": 768,
     "answer": {"backend": "typed" | "mimic" | "judge", ...}}

See ``docs/data.md`` for the ``answer`` fields of each backend.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import yaml

from ...inference.session import Session
from ...modeling.fact_writer import FactWriter
from ...modeling.model import ReMAP
from .rewards import RewardSettings, reward_function
from .rollout import TreeConfig
from .trainer import RLConfig, build_optimizer, policy_parameters, rl_content, rl_step, seed_all


@dataclass
class TrainConfig:
    checkpoint: str                       # ReMAP checkpoint (memory modules, Stage I result)
    base_model: str
    train_data: str
    output: str
    initial_policy: Optional[str] = None  # policy LoRA + request rows to start from
    epochs: int = 2
    seed: int = 20260911
    save_every: int = 8
    rl: RLConfig = field(default_factory=RLConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    reward: RewardSettings = field(default_factory=RewardSettings)
    judge_model: Optional[str] = None

    @classmethod
    def load(cls, path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text())
        rl = RLConfig(**raw.pop("rl", {}))
        tree = TreeConfig(**raw.pop("tree", {}))
        reward = raw.pop("reward", {})
        if "correct_call_bonus" in reward:
            reward["correct_call_bonus"] = tuple(reward["correct_call_bonus"])
        return cls(**raw, rl=rl, tree=tree, reward=RewardSettings(**reward))


def read_rows(path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def epoch_schedule(rows: List[dict], seed: int, epoch: int, world: int) -> List[dict]:
    """Shuffled order in which consecutive blocks of ``world`` rows form one step.

    Rows with the same generation budget and similar image counts share a step, so
    one long high-N question does not keep the other processes waiting.  The last
    block is padded with rows from the start of the order.
    """
    rng = random.Random(seed + epoch)
    ordered = list(rows)
    rng.shuffle(ordered)
    ordered.sort(key=lambda row: (int(row["max_new_tokens"]), (len(row["images"]) - 1) // 4))
    batches = [ordered[i:i + world] for i in range(0, len(ordered), world)]
    if len(batches[-1]) < world:
        batches[-1] += ordered[:world - len(batches[-1])]
    rng.shuffle(batches)
    for batch in batches:
        rng.shuffle(batch)
    return [row for batch in batches for row in batch]


def policy_state(model) -> dict:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in policy_parameters(model)}


@torch.no_grad()
def load_policy(model, state: dict) -> None:
    named = dict(policy_parameters(model))
    if set(state) != set(named):
        raise ValueError("policy checkpoint does not match the trainable policy")
    for name, parameter in named.items():
        parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def save(path: Path, model, optimizer, group: int, cfg: TrainConfig) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(dict(group=group, policy=policy_state(model), optimizer=optimizer.state_dict(),
                    config=asdict(cfg)), tmp)
    os.replace(tmp, path)


def latest_checkpoint(output: Path) -> Optional[Path]:
    found = sorted(output.glob("policy.group*.pt"), key=lambda p: int(p.stem.split("group")[1]))
    return found[-1] if found else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    log_dir = output / f"rank{rank:03d}"
    log_dir.mkdir(exist_ok=True)

    model = ReMAP.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    writer = FactWriter.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    optimizer, lora, rows = build_optimizer(model, cfg.rl)
    if cfg.initial_policy:
        load_policy(model, torch.load(cfg.initial_policy, map_location="cpu")["policy"])
    start = 0
    resume = latest_checkpoint(output)
    if resume is not None:
        state = torch.load(resume, map_location="cpu")
        load_policy(model, state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["group"])

    judge = None
    data = read_rows(cfg.train_data)
    if any(row["answer"]["backend"] == "judge" for row in data):
        from ...scoring.judge import LLMJudge
        judge = LLMJudge(model=cfg.judge_model)
    schedule = [row for epoch in range(cfg.epochs) for row in epoch_schedule(data, cfg.seed, epoch, world)]
    groups = len(schedule) // world

    for group in range(start + 1, groups + 1):
        index = (group - 1) * world + rank
        row = schedule[index]
        started = time.time()
        seed_all(cfg.seed + index)
        prompt = Session.prefill(model, row["images"], rl_content(row["question"], len(row["images"])),
                                 question=row["question"], fact_writer=writer)
        reward_fn = reward_function(model, row, (group - 1) / max(1, groups - 1), cfg.reward, judge,
                                    log_dir / "judge.jsonl")
        tree = replace(cfg.tree, max_new_tokens=int(row["max_new_tokens"]))
        stats = rl_step(model, optimizer, lora, rows, prompt, row["question"], reward_fn, tree, cfg.rl)
        stats.update(group=group, row=row["id"], seconds=time.time() - started)
        with (log_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(stats, rewards=reward_fn.records)) + "\n")
        if rank == 0 and (group % cfg.save_every == 0 or group == groups):
            save(output / f"policy.group{group}.pt", model, optimizer, group, cfg)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
