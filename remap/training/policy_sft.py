"""Supervised fine-tuning of the access policy on teacher trajectories (before Stage II).

    torchrun --nproc_per_node 8 -m remap.training.policy_sft --config configs/policy_sft.yaml

Data: JSON lines ``{"id", "images", "question", "token_ids", "loss_mask"}``.  ``token_ids``
is a complete trajectory ending with EOS (memory requests included); ``loss_mask``
marks the supervised positions.  The memory blocks of a trajectory are computed once,
by replaying it with the starting model, and then kept fixed.

Trained: the policy LoRA, the request tokens' input and output rows and the END input
rows.  Loss: mean cross-entropy over a trajectory's supervised tokens; one trajectory
per process and step; gradients averaged over processes.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from ..inference.session import Session
from ..modeling.fact_writer import FactWriter
from ..modeling.model import ReMAP, load_image
from .common import atomic_save, barrier, init_distributed, seed_all, synchronize_gradients
from .rl.scoring import build_layout, sequence_logprobs
from .rl.trainer import encode_prompt, policy_parameters, rl_content


@dataclass
class PolicySFTConfig:
    checkpoint: str
    base_model: str
    train_data: str
    output: str
    steps: int
    initial_policy: str = ""
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_calls: int = 4
    seed: int = 20260911
    save_every: int = 16
    zero_lora_b: bool = False          # start the policy LoRA from zero B matrices (keeps A)

    @classmethod
    def load(cls, path) -> "PolicySFTConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


def trainable(model):
    return policy_parameters(model) + [("memory_rows.embed_end", model.memory_rows.embed_end)]


class MemoryCache:
    """Memory blocks of each trajectory, replayed once with the starting model."""

    def __init__(self, model, writer, folder: Path, max_calls: int):
        self.model, self.writer, self.folder, self.max_calls = model, writer, folder, max_calls
        folder.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(self, row):
        path = self.folder / f"{row['id'].replace(':', '_')}.pt"
        if path.exists():
            return torch.load(path, map_location=self.model.device)
        prompt = Session.prefill(self.model, row["images"], rl_content(row["question"], len(row["images"])),
                                 question=row["question"], fact_writer=self.writer)
        session = Session(self.model, prompt)
        session.replay(row["token_ids"][:-1])
        reader = next((e.reader for e in reversed(session.events) if e.kind == "local"), None)
        value = dict(blocks=[e.block.detach() for e in session.events], reader=reader)
        torch.save(value, path)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = PolicySFTConfig.load(parser.parse_args().config)
    rank, world = init_distributed()
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    model = ReMAP.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    writer = FactWriter.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    named = trainable(model)
    if cfg.initial_policy:
        from .rl.train import load_policy
        load_policy(model, torch.load(cfg.initial_policy, map_location="cpu")["policy"])
    model.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        for name, parameter in named:
            parameter.data = parameter.data.float()
            if cfg.zero_lora_b and ".lora_B.policy." in name:
                parameter.zero_()
            parameter.requires_grad_(True)
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rows = [json.loads(line) for line in Path(cfg.train_data).read_text().splitlines() if line.strip()]
    cache = MemoryCache(model, writer, output / f"memory_cache.rank{rank:03d}", cfg.max_calls)
    rng, order, epoch = random.Random(cfg.seed), [], -1
    log = (output / f"rank{rank:03d}.jsonl").open("a")
    for step in range(1, cfg.steps + 1):
        slot = (step - 1) * world + rank
        wanted, position = divmod(slot, len(rows))
        while epoch < wanted:              # every process walks the same sequence of permutations
            order = list(range(len(rows)))
            rng.shuffle(order)
            epoch += 1
        row = rows[order[position]]
        memory = cache(row)
        images = [load_image(path) for path in row["images"]]
        prompt_ids, pixel_values, grid = encode_prompt(model, images, row["question"])
        layout = build_layout(prompt_ids, row["token_ids"], model.token_ids)
        optimizer.zero_grad(set_to_none=True)
        logp = sequence_logprobs(model, layout, memory["blocks"], pixel_values, grid,
                                 reader_memory=memory["reader"], max_calls=cfg.max_calls)
        mask = torch.tensor(row["loss_mask"], dtype=torch.bool, device=logp.device)
        loss = -logp[mask].mean()
        loss.backward()
        synchronize_gradients(params, world)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        log.write(json.dumps(dict(step=step, row=row["id"], loss=float(loss.detach()), grad_norm=float(norm))) + "\n")
        log.flush()
        if rank == 0 and (step % cfg.save_every == 0 or step == cfg.steps):
            atomic_save(dict(step=step, policy={n: p.detach().cpu().clone() for n, p in policy_parameters(model)},
                             embed_end=model.memory_rows.embed_end.detach().cpu().clone(),
                             optimizer=optimizer.state_dict(), config=asdict(cfg)), output / f"policy.step{step}.pt")
        barrier()


if __name__ == "__main__":
    main()
