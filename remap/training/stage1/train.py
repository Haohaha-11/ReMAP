"""Stage I training of one memory component.

    torchrun --nproc_per_node 8 -m remap.training.stage1.train --config configs/stage1_global.yaml

One state per process and step; gradients are averaged over processes.  The data is a
JSON-lines manifest of ``{"state": <state file>, "target": <target file>}`` (written by
``stage1.annotate`` or ``stage1.review_roi``).  Components:

* ``global``: the whole Global memory (weaver, clause writer, bridges, readout);
  suffix cross-entropy after a Global read.
* ``local``: the Local weaver's query/cross-attention core and its crop read;
  suffix cross-entropy after a Local read.
* ``twig``: Twig's LoRA (the part that produces the score map); ROI loss on the
  reviewed boxes of a Local read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from ...modeling.model import ReMAP
from ..common import atomic_save, barrier, epoch_slots, init_distributed, seed_all, synchronize_gradients
from .objective import suffix_loss
from .roi import locator_loss
from .states import MemoryState, load_state

LOCAL_TRAINABLE = ("local_queries", "query_cond_proj.", "state_kv_proj.", "bbox_proj.", "layers.", "out_ln.",
                   "context_crop_")


@dataclass
class Stage1Config:
    component: str                     # "global" | "local" | "twig"
    checkpoint: str
    base_model: str
    train_data: str
    output: str
    epochs: int
    learning_rate: float
    seed: int = 20260920
    start_epoch: int = 0               # resume a schedule; earlier epochs' slots are skipped
    init: Optional[str] = None         # component weights to start from (fresh optimizer)
    max_calls: int = 4
    grad_clip: float = 1.0

    @classmethod
    def load(cls, path) -> "Stage1Config":
        return cls(**yaml.safe_load(Path(path).read_text()))


def trainable_parameters(model, component: str):
    if component == "global":
        return [("global_memory." + n, p) for n, p in model.global_memory.named_parameters()]
    if component == "local":
        return [("local_memory.weaver." + n, p) for n, p in model.local_memory.weaver.named_parameters()
                if n.startswith(LOCAL_TRAINABLE)]
    if component == "twig":
        last = len(model.twig.layers) - 1
        # Only the last layer's query/key projections reach the score map.
        return [("twig." + n, p) for n, p in model.twig.named_parameters() if ".locator." in n
                and (not n.startswith(f"layers.{last}.") or ".q_proj." in n or ".k_proj." in n)]
    raise ValueError(f"unknown component: {component}")


def twig_loss(model, state: MemoryState, targets):
    """ROI loss of the Twig score maps at the Local read of ``state``."""
    length = int(state.inputs["query_length"])
    embeds, positions = state.embeds[:, :length], state.positions[..., :length]
    attention = torch.ones((1, length), dtype=torch.long, device=embeds.device)
    with torch.no_grad():
        hidden = model.base.model(inputs_embeds=embeds, position_ids=positions, attention_mask=attention,
                                  use_cache=False, output_hidden_states=True).hidden_states[model.config.twig_branch_layer]
    heats = model.twig.image_heats(model.base.model, hidden, positions, length - 1,
                                   state.image_spans(model.base.config.image_token_id), attention_mask=attention)
    return locator_loss(heats, targets)


def component_state(named) -> dict:
    return {name: p.detach().cpu().clone() for name, p in named}


@torch.no_grad()
def load_component(named, state: dict) -> None:
    by_name = dict(named)
    if set(state) != set(by_name):
        raise ValueError("component checkpoint does not match the trainable parameters")
    for name, parameter in by_name.items():
        parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty training manifest")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = Stage1Config.load(parser.parse_args().config)
    rank, world = init_distributed()
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)

    model = ReMAP.from_pretrained(cfg.checkpoint, base_model=cfg.base_model)
    named = trainable_parameters(model, cfg.component)
    model.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        for _, parameter in named:
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
    if cfg.init:
        load_component(named, torch.load(cfg.init, map_location="cpu")["state"])
    params = [p for _, p in named]
    seed_all(cfg.seed + rank)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    rows = read_manifest(cfg.train_data)
    indices = epoch_slots(len(rows), world, cfg.epochs, cfg.seed)
    steps_per_epoch = math.ceil(len(rows) / world)
    log = (output / f"rank{rank:03d}.jsonl").open("a")
    for step in range(cfg.start_epoch * steps_per_epoch + 1, cfg.epochs * steps_per_epoch + 1):
        started = time.time()
        slot = (step - 1) * world + rank
        seed_all(cfg.seed + slot)
        row = rows[indices[slot]]
        state = load_state(row["state"], model.device)
        target = json.loads(Path(row["target"]).read_text())
        optimizer.zero_grad(set_to_none=True)
        if cfg.component == "twig":
            loss, tokens = twig_loss(model, state, target["targets"]), 0
        else:
            loss, tokens = suffix_loss(model, state, target["suffix_text"], target["endpoint"],
                                       max_calls=cfg.max_calls)
        loss.backward()
        synchronize_gradients(params, world)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        log.write(json.dumps(dict(step=step, state=row["state"], loss=float(loss.detach()), tokens=tokens,
                                  grad_norm=float(norm), seconds=time.time() - started)) + "\n")
        log.flush()
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            if rank == 0:
                atomic_save(dict(component=cfg.component, epoch=epoch, step=step, state=component_state(named),
                                 optimizer=optimizer.state_dict(), config=asdict(cfg)),
                            output / f"{cfg.component}.epoch{epoch}.pt")
            barrier()


if __name__ == "__main__":
    main()
