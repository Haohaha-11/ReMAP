"""Twig ROI objective: teach the score map to cover a reviewed region.

For each image with a reviewed box, the target is the fraction of every score-map
cell covered by the box.  The loss combines

* class-balanced BCE between the cell logits and the coverage,
* 1 - soft IoU,
* a *readout* term matching the deployed box rule (cells with probability at least
  half the peak are kept): every required cell should clear that threshold and every
  other cell should stay below the weakest required one.

Several boxes on one image are merged into their enclosing box, since Local makes a
single crop per image.  Images without a reviewed box get no loss.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def coverage_target(box, shape, *, device) -> torch.Tensor:
    """Per-cell coverage of a normalised ``box`` on a ``shape`` = (rows, cols) grid (fp64 geometry)."""
    x0, y0, x1, y1 = map(float, box)
    rows, cols = map(int, shape)
    if not (rows > 0 and cols > 0 and 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("invalid normalised box or score map")
    x = torch.arange(cols + 1, device=device, dtype=torch.float64) / cols
    y = torch.arange(rows + 1, device=device, dtype=torch.float64) / rows
    overlap_x = (torch.minimum(x[1:], x.new_tensor(x1)) - torch.maximum(x[:-1], x.new_tensor(x0))).clamp_min(0) * cols
    overlap_y = (torch.minimum(y[1:], y.new_tensor(y1)) - torch.maximum(y[:-1], y.new_tensor(y0))).clamp_min(0) * rows
    target = (overlap_y[:, None] * overlap_x[None, :]).clamp(0, 1).float()
    if not bool(target.max() > 0):
        raise ValueError("box covers no cell")
    return target


def box_loss(heat: torch.Tensor, box) -> torch.Tensor:
    target = coverage_target(box, heat.shape, device=heat.device)
    logits = heat.float()
    weight = ((target.numel() - target.sum()) / target.sum()).clamp(max=50.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=1.0 + (weight - 1.0) * target)
    p = logits.sigmoid()
    intersection = (p * target).sum()
    soft_iou = intersection / (p.sum() + target.sum() - intersection).clamp_min(1e-6)
    support = target >= 0.5 * target.max()
    inside, outside = logits[support], logits[~support]
    inside_floor = F.softplus(logits.max() + math.log(0.5) - inside).mean()
    outside_ceiling = (F.softplus(outside.max() - inside.min() - math.log(0.5)) if outside.numel()
                       else logits.sum() * 0.0)
    return bce + 1.0 - soft_iou + (inside_floor + outside_ceiling) / 2


def enclosing_boxes(targets: Sequence[dict], n_images: int) -> Dict[int, List[float]]:
    """{image_index: enclosing normalised box} of reviewed targets."""
    groups = {}
    for target in targets:
        index = int(target["image_index"])
        if not 0 <= index < n_images:
            raise ValueError("target image outside the question's images")
        groups.setdefault(index, []).append(tuple(map(float, target["bbox_normalized"])))
    if not groups:
        raise ValueError("no reviewed box")
    return {i: [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
            for i, bs in groups.items()}


def locator_loss(heats: Sequence[torch.Tensor], targets: Sequence[dict]) -> torch.Tensor:
    boxes = enclosing_boxes(targets, len(heats))
    return torch.stack([box_loss(heats[i], box) for i, box in sorted(boxes.items())]).mean()
