"""Stage I objective: suffix cross-entropy after one observed memory read.

The student's own decoder input up to a request is kept fixed (:class:`MemoryState`).
The block of the request is recomputed by the module being trained and inserted as
``[request; block; END]``; a teacher-written continuation of the student's reasoning
follows.  The loss is the mean cross-entropy of the continuation tokens, plus EOS when
the continuation ends with the final answer.  The decoder is frozen and only passes
gradients through to the inserted block (and to the reader memory of a Local read).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from ...inference.session import answer_committed
from .states import MemoryState


def suffix_targets(tokenizer, token_ids, suffix_text: str, endpoint: str) -> List[int]:
    ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    eos = int(tokenizer.eos_token_id)
    if not ids or set(ids) & set(token_ids.all_ids) or eos in ids:
        raise ValueError("continuation is empty or contains protocol tokens")
    if endpoint not in ("final_answer", "partial"):
        raise ValueError(f"unknown continuation endpoint: {endpoint}")
    return ids + [eos] if endpoint == "final_answer" else ids


def recompute_block(model, state: MemoryState):
    """(block, reader) of the state's request; differentiable in the memory modules."""
    if state.kind == "global":
        return model.weave_global(state.global_inputs(model.base.config.image_token_id)), None
    image_states = state.image_states(model.base.config.image_token_id)
    return model.weave_local(state.local_inputs(), image_states, state.relation_tokens)


def suffix_loss(model, state: MemoryState, suffix_text: str, endpoint: str, *, block=None, reader=None,
                max_calls: int = 4):
    """Mean continuation cross-entropy (scalar) and the number of supervised tokens."""
    tokenizer, token_ids = model.processor.tokenizer, model.token_ids
    if block is None:
        block, reader = recompute_block(model, state)
    targets = suffix_targets(tokenizer, token_ids, suffix_text, endpoint)
    calls = [k for k in map(token_ids.kind, state.prefix_ids) if k is not None] + [state.kind]
    if len(calls) > max_calls:
        raise ValueError("the request exceeds the call budget")
    device = model.device
    dtype = state.embeds.dtype
    request = token_ids.mem_g if state.kind == "global" else token_ids.mem_l
    pieces, positions = [state.embeds.to(device)], [state.positions.to(device)]
    length, position = state.embeds.shape[1], state.next_position

    def append(value):
        nonlocal length, position
        width = value.shape[1]
        pieces.append(value)
        positions.append(torch.arange(position, position + width, device=device).view(1, 1, -1).expand(3, 1, -1))
        length, position = length + width, position + width

    embed = lambda token: model.embed(torch.tensor([[token]], device=device))  # noqa: E731
    append(embed(request))
    append(torch.cat([block.to(dtype), embed(token_ids.end_for(request))], dim=1))
    predictors, generated, answered = [], list(state.prefix_ids) + [request], False
    for index, token in enumerate(targets):
        predictors.append((length - 1, answered))
        if index + 1 < len(targets):
            append(embed(token))
        generated.append(token)
        answered = answered or answer_committed(tokenizer, generated)
    if endpoint == "final_answer" and not answered:
        raise ValueError("a final-answer continuation must contain 'Answer:'")
    sequence = torch.cat(pieces, dim=1)
    hidden = model.base.model(inputs_embeds=sequence, position_ids=torch.cat(positions, dim=-1),
                              attention_mask=torch.ones(sequence.shape[:2], dtype=torch.long, device=device),
                              past_key_values=None, use_cache=False).last_hidden_state
    reader_memory: Optional[torch.Tensor] = reader if state.kind == "local" else state.previous_reader
    losses = []
    for token, (offset, active) in zip(targets, predictors):
        h = hidden[:, offset:offset + 1]
        if "local" in calls and active:
            h = model.read_answer_hidden(h, reader_memory)
        logits = model.logits(h).float().reshape(1, -1).clone()
        logits[:, token_ids.end_ids] = -torch.inf
        if active or len(calls) >= max_calls:
            logits[:, token_ids.request_ids] = -torch.inf
        losses.append(F.cross_entropy(logits, torch.tensor([token], device=device)))
    return torch.stack(losses).mean(), len(losses)
