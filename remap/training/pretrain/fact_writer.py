"""Fact writer training: weighted SFT on written visual facts.

    torchrun --nproc_per_node 8 -m remap.training.pretrain.fact_writer --config configs/pretrain_fact_writer.yaml

The fact writer (:class:`remap.modeling.fact_writer.FactWriter`) is the base model with a
decoder LoRA and a vision LoRA; given 1-3 images and the question it writes
``I1{...}; ...; REL{...}.``.  Training rows: ``{"id", "images", "question", "fact",
"weight"}``.  Per row the loss is

    weight * (weighted NLL of fact + EOS  +  anchor_weight * mean_p mean((p - p0)^2))

where schema characters (labels, braces, separators) weigh ``structure_weight``, EOS
weighs ``eos_weight``, other tokens 1, and ``p0`` are the LoRA values at the start.  The
released writer ran this on its own free-generation errors (weight 2) mixed with
correct anchors (weight 1), starting from the previous writer.

Batches: process ``r`` takes rows ``r, r + world, ...`` in file order; gradients are
accumulated over ``grad_accum`` rows per process and averaged over processes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image

from ...modeling.fact_writer import instruction, min_side
from ..common import init_distributed


@dataclass
class FactWriterConfig:
    base_model: str
    init: str                         # fact-writer checkpoint directory to start from ("" for fresh LoRA)
    train_data: str
    output: str
    learning_rate: float = 1e-6
    grad_accum: int = 4
    structure_weight: float = 3.0
    eos_weight: float = 4.0
    anchor_weight: float = 0.01
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    save_every: int = 30
    seed: int = 20260923

    @classmethod
    def load(cls, path) -> "FactWriterConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


def structural_mask(fact: str):
    """Schema characters are structural; text inside braces is content."""
    mask, inside = [], False
    for character in fact:
        if character == "{":
            mask.append(True)
            inside = True
        elif character == "}":
            mask.append(True)
            inside = False
        else:
            mask.append(not inside)
    return mask


def encode(processor, image_paths, question: str, fact: str, structure_weight: float, eos_weight: float):
    images = [min_side(Image.open(path).convert("RGB")) for path in image_paths]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction(len(images), question)}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True)
    fact = fact.strip()
    encoded = processor(text=[prompt + fact + processor.tokenizer.eos_token], images=images, return_tensors="pt")
    prompt_length = int(processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1])
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_length] = -100
    tokens = processor.tokenizer(fact, add_special_tokens=False, return_offsets_mapping=True)
    mask = structural_mask(fact)
    weights = [structure_weight if any(mask[i] for i in range(a, b)) else 1.0 for a, b in tokens["offset_mapping"]]
    weights.append(eos_weight)
    if labels[labels != -100].tolist() != list(tokens["input_ids"]) + [processor.tokenizer.eos_token_id]:
        raise ValueError("prompt/target token boundary differs")
    token_weights = torch.zeros_like(labels, dtype=torch.float32)
    token_weights[labels != -100] = torch.tensor(weights)
    return encoded, labels, token_weights


def weighted_nll(logits, labels, weights):
    shifted, targets = logits[..., :-1, :], labels[..., 1:]
    mask = targets != -100
    losses = F.cross_entropy(shifted[mask].float(), targets[mask], reduction="none")
    w = weights[..., 1:][mask].to(losses.device, losses.dtype)
    return (losses * w).sum() / w.sum()


def lora_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if ".writer." in n or ".writer_vision." in n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = FactWriterConfig.load(parser.parse_args().config)
    rank, world = init_distributed()
    torch.manual_seed(cfg.seed)
    from transformers import AutoProcessor

    from ...modeling.fact_writer import FactWriter
    writer = FactWriter.from_pretrained(cfg.init, base_model=cfg.base_model)
    model = writer.model
    processor = AutoProcessor.from_pretrained(cfg.base_model, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
                                              use_fast=True)
    processor.chat_template = writer.processor.chat_template
    model.requires_grad_(False)
    named = lora_parameters(model)
    with torch.no_grad():
        for _, parameter in named:
            parameter.data = parameter.data.to(torch.bfloat16)
            parameter.requires_grad_(True)
    params = [p for _, p in named]
    anchors = [p.detach().clone() for p in params]
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    rows = [json.loads(line) for line in Path(cfg.train_data).read_text().splitlines() if line.strip()]
    block = world * cfg.grad_accum
    rows += [dict(rows[i % len(rows)], weight=0.0) for i in range((-len(rows)) % block)]   # zero-weight padding
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / f"rank{rank:03d}.jsonl").open("a")
    steps = len(rows) // block
    for step in range(1, steps + 1):
        for micro in range(cfg.grad_accum):
            row = rows[((step - 1) * cfg.grad_accum + micro) * world + rank]
            inputs, labels, weights = encode(processor, row["images"], row["question"], row["fact"],
                                             cfg.structure_weight, cfg.eos_weight)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs, use_cache=False).logits
            nll = weighted_nll(logits, labels.to(model.device), weights.to(model.device))
            anchor = torch.stack([(p - a).float().square().mean() for p, a in zip(params, anchors)]).mean()
            loss = float(row["weight"]) * (nll + cfg.anchor_weight * anchor)
            (loss / cfg.grad_accum).backward()
            log.write(json.dumps(dict(step=step, id=row["id"], nll=float(nll.detach()), weight=row["weight"])) + "\n")
        for parameter in params:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if world > 1:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log.flush()
        if rank == 0 and (step % cfg.save_every == 0 or step == steps):
            from safetensors.torch import save_file
            folder = output / f"step{step}"
            folder.mkdir(exist_ok=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in named}, str(folder / "fact_writer.safetensors"))
            for name in ("fact_writer_config.json", "chat_template.json"):
                source = Path(cfg.init) / name
                if source.exists():
                    (folder / name).write_text(source.read_text())
            (folder / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1))
        if world > 1:
            dist.barrier()


if __name__ == "__main__":
    main()
