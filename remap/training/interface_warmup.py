"""Interface warm-up before Stage II: train the six memory-token rows on complete trajectories.

    torchrun --nproc_per_node 16 -m remap.training.interface_warmup --config configs/interface_warmup.yaml

Data: successful (correct, EOS-terminated) trajectories of the current model, both with
several memory reads and with none, as JSON lines
``{"id", "images", "question", "token_ids"}`` (``token_ids`` ends with EOS).

Before every use a trajectory is replayed with the current model, so its memory blocks
reflect the current rows.  Every generated token is a target, in one of three groups:
``call`` (request tokens), ``resume`` (ordinary tokens after the first read) and
``no_call`` (ordinary tokens before any read).  The loss is the sum over groups of the
group's summed cross-entropy divided by the group's token count in the whole global
batch; gradients are summed over processes.  Only the input rows of the request and
END tokens and the output rows of the request tokens are trained.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..inference.session import Session
from ..modeling.fact_writer import FactWriter
from ..modeling.model import ReMAP
from .common import atomic_save, barrier, init_distributed, seed_all
from .rl.scoring import build_layout, sequence_logprobs
from .rl.trainer import encode_prompt, rl_content

GROUPS = ("call", "resume", "no_call")


@dataclass
class WarmupConfig:
    checkpoint: str
    base_model: str
    train_data: str
    output: str
    initial_policy: str = ""           # policy LoRA + request rows the warm-up starts from
    epochs: int = 3
    batch_size: int = 16               # global batch; one trajectory per process at a time
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    max_calls: int = 4
    seed: int = 20260920

    @classmethod
    def load(cls, path) -> "WarmupConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


def target_groups(token_ids, ids) -> dict:
    groups, read = {g: [] for g in GROUPS}, False
    for index, token in enumerate(ids):
        if token_ids.kind(token) is not None:
            groups["call"].append(index)
            read = True
        else:
            groups["resume" if read else "no_call"].append(index)
    return groups


def interface_rows(model):
    rows = model.memory_rows
    return [("memory_rows.embed_request", rows.embed_request), ("memory_rows.embed_end", rows.embed_end),
            ("memory_rows.head_request", rows.head_request)]


def trajectory_logprobs(model, writer, row, max_calls: int):
    """Log-probabilities of ``row["token_ids"]`` with freshly replayed memory."""
    ids = list(row["token_ids"])
    with torch.no_grad():
        prompt = Session.prefill(model, row["images"], rl_content(row["question"], len(row["images"])),
                                 question=row["question"], fact_writer=writer)
        session = Session(model, prompt)
        session.replay(ids[:-1])
    prompt_ids, pixel_values, grid = encode_prompt(model, prompt.state.images, row["question"])
    layout = build_layout(prompt_ids, ids, model.token_ids)
    reader = next((e.reader for e in reversed(session.events) if e.kind == "local"), None)
    return sequence_logprobs(model, layout, [e.block for e in session.events], pixel_values, grid,
                             reader_memory=reader, max_calls=max_calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = WarmupConfig.load(parser.parse_args().config)
    rank, world = init_distributed()
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    model = ReMAP.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    writer = FactWriter.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    if cfg.initial_policy:
        from .rl.train import load_policy
        load_policy(model, torch.load(cfg.initial_policy, map_location="cpu")["policy"])
    model.requires_grad_(False)
    model.eval()
    named = interface_rows(model)
    with torch.no_grad():
        for _, parameter in named:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    data = [json.loads(line) for line in Path(cfg.train_data).read_text().splitlines() if line.strip()]
    groups = [target_groups(model.token_ids, row["token_ids"]) for row in data]
    log = (output / f"rank{rank:03d}.jsonl").open("a")
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(data)))
        random.Random(cfg.seed + epoch - 1).shuffle(order)
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            counts = {g: sum(len(groups[i][g]) for i in batch) for g in GROUPS}
            optimizer.zero_grad(set_to_none=True)
            sums = {g: 0.0 for g in GROUPS}
            for index in batch[rank::world]:
                seed_all(cfg.seed + step * cfg.batch_size + index)
                logp = trajectory_logprobs(model, writer, data[index], cfg.max_calls)
                terms = []
                for g in GROUPS:
                    if groups[index][g]:
                        total = -logp[torch.tensor(groups[index][g], device=logp.device)].sum()
                        terms.append(total / counts[g])
                        sums[g] += float(total.detach())
                torch.stack(terms).sum().backward()
            present = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=model.device)
            if world > 1:
                dist.all_reduce(present)
            for parameter, used in zip(params, present.tolist()):
                if not used:              # no process used this row: Adam skips it
                    parameter.grad = None
                    continue
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                if world > 1:             # summed: the loss already uses global token counts
                    dist.all_reduce(parameter.grad)
            params_with_grad = [p for p in params if p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(params_with_grad, cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            log.write(json.dumps(dict(step=step, epoch=epoch, counts=counts, sums=sums, grad_norm=float(norm))) + "\n")
            log.flush()
        if rank == 0:
            atomic_save(dict(epoch=epoch, step=step, state={n: p.detach().cpu().clone() for n, p in named},
                             optimizer=optimizer.state_dict(), config=asdict(cfg)),
                        output / f"interface.epoch{epoch}.pt")
        barrier()


if __name__ == "__main__":
    main()
