"""Training states for Stage I: a student trajectory stopped at one memory request.

A state file (``torch.save``) holds what the student actually saw up to the request
and the frozen inputs of the requested block:

    kind             "global" | "local"
    question, images the public question and image paths
    prompt_ids       [L_p] prompt token ids;  grid  [N, 3] image grids
    prefix_ids       generated token ids before the request (earlier requests included)
    embeds           [1, L, d] decoder inputs so far (prompt, tokens, earlier memory blocks)
    positions        [3, 1, L] their rotary positions;  next_position  position of the request
    previous_reader  [1, 32, d] reader memory of the latest earlier Local request, or None
    relation_tokens  [1, 16, d] relation memory of the prompt
    inputs           the requested block's frozen inputs (see :func:`pack_inputs`)

Image features are not stored separately: they are the prompt embeddings at the image
token positions.  A target file (JSON) adds the teacher continuation for the state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from ...modeling.model import GlobalInputs, LocalInputs, image_inputs


@dataclass
class MemoryState:
    kind: str
    question: str
    images: List[str]
    prompt_ids: torch.Tensor
    grid: torch.Tensor
    prefix_ids: List[int]
    embeds: torch.Tensor
    positions: torch.Tensor
    next_position: int
    relation_tokens: torch.Tensor
    inputs: dict
    previous_reader: Optional[torch.Tensor] = None
    state_id: str = ""

    def image_states(self, image_token_id: int) -> List[torch.Tensor]:
        """Per-image visual features [1, L_i, d], recovered from the prompt embeddings."""
        positions = (self.prompt_ids == image_token_id).nonzero(as_tuple=True)[0].to(self.embeds.device)
        features = self.embeds[0, positions]
        states, offset = [], 0
        for t, h, w in self.grid.tolist():
            count = t * h * w // 4
            states.append(features[offset:offset + count].unsqueeze(0))
            offset += count
        if offset != features.shape[0]:
            raise ValueError("image grid does not match the image tokens of the prompt")
        return states

    def global_inputs(self, image_token_id: int) -> GlobalInputs:
        image_states = self.image_states(image_token_id)
        visual, image_ids = image_inputs(image_states)
        count = len(image_states)
        if count <= 3:
            return GlobalInputs(count, visual, image_ids, clauses=self.inputs["clauses"])
        summaries = torch.stack([s[0].float().mean(dim=0) for s in image_states]).to(dtype=torch.bfloat16)
        return GlobalInputs(count, visual, image_ids, summaries=summaries, question=self.inputs["question"],
                            request_state=self.inputs["request_state"])

    def local_inputs(self) -> LocalInputs:
        x = self.inputs
        return LocalInputs(tuple(x["positions"]), list(x["regions"]), list(x["bboxes"]), x["query_state"],
                           list(x.get("boxes", [])), int(x.get("query_length", 0)))

    def image_spans(self, image_token_id: int):
        """[(visual token indices in the prompt, (rows, cols))] per image."""
        positions = (self.prompt_ids == image_token_id).nonzero(as_tuple=True)[0]
        spans, offset = [], 0
        for t, h, w in self.grid.tolist():
            count = t * h * w // 4
            spans.append((positions[offset:offset + count], (h // 2, w // 2)))
            offset += count
        return spans

    def to(self, device) -> "MemoryState":
        def move(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(v) for v in value)
            return value
        return MemoryState(**{k: move(v) for k, v in self.__dict__.items()})


def pack_inputs(inputs) -> dict:
    """Frozen block inputs to store (visual features are recovered from the embeddings)."""
    cpu = lambda t: t.detach().cpu()  # noqa: E731
    if isinstance(inputs, LocalInputs):
        return dict(positions=list(inputs.positions), regions=[cpu(r) for r in inputs.regions],
                    bboxes=[cpu(b) for b in inputs.bboxes], query_state=cpu(inputs.query_state),
                    boxes=[list(map(float, b)) for b in inputs.boxes], query_length=int(inputs.query_length))
    if inputs.image_count <= 3:
        return dict(clauses=cpu(inputs.clauses))
    return dict(question=cpu(inputs.question), request_state=cpu(inputs.request_state))


def save_state(path, state: MemoryState) -> None:
    value = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.__dict__.items()}
    torch.save(value, Path(path))


def load_state(path, device="cpu") -> MemoryState:
    return MemoryState(**torch.load(Path(path), map_location="cpu", weights_only=False)).to(device)
