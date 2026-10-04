"""Utilities shared by the training stages."""
from __future__ import annotations

import math
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.distributed as dist


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def init_distributed():
    """(rank, world); one process per GPU, launched by torchrun or the cluster."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    return rank, world


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def epoch_slots(count: int, world: int, epochs: int, seed: int) -> List[int]:
    """Sample indices for every (step, rank) slot; slot ``(step - 1) * world + rank``.

    Each epoch is a fresh permutation (``random.Random(seed + epoch)``) of all samples,
    repeated cyclically to fill ``ceil(count / world)`` steps.
    """
    steps = math.ceil(count / world)
    indices = []
    for epoch in range(epochs):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        slots = steps * world
        indices += (order * math.ceil(slots / count))[:slots]
    return indices


def synchronize_gradients(parameters: Sequence[torch.nn.Parameter], world: int) -> List[int]:
    """Average gradients over ranks.  A parameter no rank used keeps ``grad=None`` (Adam skips it)."""
    counts = []
    for parameter in parameters:
        active = torch.tensor(int(parameter.grad is not None), device=parameter.device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(active)
        count = int(active.item())
        counts.append(count)
        if count == 0:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if world > 1:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world)
    return counts


def atomic_save(value, path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)
