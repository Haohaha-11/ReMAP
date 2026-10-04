"""Teacher-forced log-probabilities of a sampled trajectory, with its memory blocks re-inserted.

A trajectory is the prompt followed by the sampled tokens.  Each request token is
followed by its END token in the discrete stream, and its latent block (kept from the
rollout; memory modules are frozen in Stage II) is spliced between the two.  One
forward pass then gives the policy log-probability of every sampled token,
including a sampled EOS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

_ANSWER = re.compile(r"[Aa]nswer\s*:")


@dataclass
class Layout:
    ids: List[int]               # discrete stream: prompt + generated tokens, END after each request
    response: List[bool]         # True at sampled (policy) tokens
    insert_after: List[int]      # position of each request token
    kinds: List[str]             # 'global' / 'local' per request
    generated_start: int


def build_layout(prompt_ids: Sequence[int], generated: Sequence[int], token_ids) -> Layout:
    ids, response, insert_after, kinds = list(map(int, prompt_ids)), [False] * len(prompt_ids), [], []
    for token in generated:
        ids.append(int(token))
        response.append(True)
        kind = token_ids.kind(int(token))
        if kind is not None:
            insert_after.append(len(ids) - 1)
            kinds.append(kind)
            ids.append(token_ids.end_for(int(token)))
            response.append(False)
    return Layout(ids, response, insert_after, kinds, len(prompt_ids))


def splice(values: Sequence, insert_after: Sequence[int], widths: Sequence[int], fill) -> list:
    """Insert ``widths[i]`` copies of ``fill`` right after ``insert_after[i]``."""
    out, prev = [], 0
    for anchor, width in sorted(zip(insert_after, widths)):
        out.extend(values[prev:anchor + 1])
        out.extend([fill] * int(width))
        prev = anchor + 1
    out.extend(values[prev:])
    return out


def splice_embeddings(embeds, attention_mask, positions, blocks, insert_after):
    """Insert latent blocks into a batch-1 sequence; positions continue without gaps."""
    if not blocks:
        return embeds, attention_mask, positions
    order = sorted(range(len(blocks)), key=lambda i: insert_after[i])
    parts, masks, prev = [], [], 0
    for i in order:
        anchor = int(insert_after[i]) + 1
        block = blocks[i].to(embeds.dtype)
        parts += [embeds[:, prev:anchor], block]
        masks += [attention_mask[:, prev:anchor],
                  torch.ones(1, block.shape[1], dtype=attention_mask.dtype, device=embeds.device)]
        prev = anchor
    parts.append(embeds[:, prev:])
    masks.append(attention_mask[:, prev:])
    offset = torch.zeros(positions.shape[-1], dtype=positions.dtype, device=positions.device)
    for i in order:
        offset[int(insert_after[i]) + 1:] += int(blocks[i].shape[1])
    shifted = positions + offset
    position_parts, prev = [], 0
    for i in order:
        anchor = int(insert_after[i]) + 1
        steps = torch.arange(1, int(blocks[i].shape[1]) + 1, device=positions.device)
        position_parts += [shifted[..., prev:anchor], shifted[..., anchor - 1].unsqueeze(-1) + steps]
        prev = anchor
    position_parts.append(shifted[..., prev:])
    return torch.cat(parts, dim=1), torch.cat(masks, dim=1), torch.cat(position_parts, dim=-1)


def _support_mask(layout: Layout, token_ids, tokenizer, max_calls: int) -> List[bool]:
    """Positions where request tokens were not allowed: call budget used up or answer begun."""
    blocked = [False] * len(layout.ids)
    generated, calls, answering = [], 0, False
    for index in range(layout.generated_start, len(layout.ids)):
        token = int(layout.ids[index])
        if token not in token_ids.end_ids:
            generated.append(token)
            calls += int(token_ids.kind(token) is not None)
            answering = answering or bool(_ANSWER.search(tokenizer.decode(generated[-8:], skip_special_tokens=True)))
        blocked[index] = calls >= max_calls or answering
    return blocked


def _reader_mask(layout: Layout, token_ids, tokenizer) -> List[bool]:
    """Positions after a Local request whose answer has begun (reader memory applies)."""
    active, generated = [False] * len(layout.ids), []
    local_seen = answer_seen = False
    for index in range(layout.generated_start, len(layout.ids)):
        token = int(layout.ids[index])
        if token not in token_ids.end_ids:
            generated.append(token)
            local_seen = local_seen or token == token_ids.mem_l
            answer_seen = answer_seen or bool(_ANSWER.search(tokenizer.decode(generated[-8:], skip_special_tokens=True)))
        active[index] = local_seen and answer_seen
    return active


def sequence_logprobs(model, layout: Layout, blocks: Sequence[torch.Tensor], pixel_values, image_grid_thw,
                      *, reader_memory: Optional[torch.Tensor], max_calls: int = 5) -> torch.Tensor:
    """Log-probabilities (fp32) of the sampled tokens of ``layout``, in order."""
    tokenizer = model.processor.tokenizer
    token_ids = model.token_ids
    widths = [int(b.shape[1]) for b in blocks]
    ids = splice(layout.ids, layout.insert_after, widths, 0)
    response = splice(layout.response, layout.insert_after, widths, False)
    targets_at = [p for p, selected in enumerate(response) if selected and p >= 1]
    predictors = [p - 1 for p in targets_at]
    targets = [ids[p] for p in targets_at]

    input_ids = torch.tensor([layout.ids], device=model.device, dtype=torch.long)
    embeds = model.embed(input_ids)
    image_embeds = model.base.visual(pixel_values.type(model.base.visual.dtype), grid_thw=image_grid_thw)
    image_embeds = image_embeds.to(embeds.device, embeds.dtype)
    mask = (input_ids == model.base.config.image_token_id).unsqueeze(-1).expand_as(embeds)
    embeds = embeds.masked_scatter(mask, image_embeds)
    attention = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    positions, _ = model.base.get_rope_index(input_ids, image_grid_thw, None, None, attention)
    embeds, attention, positions = splice_embeddings(
        embeds, attention, positions, [b.to(model.device) for b in blocks], layout.insert_after)
    hidden = model.base.model(input_ids=None, position_ids=positions, attention_mask=attention,
                              past_key_values=None, inputs_embeds=embeds, use_cache=False,
                              output_hidden_states=False, return_dict=True, cache_position=None)[0]
    if "local" in layout.kinds:
        active = torch.tensor(splice(_reader_mask(layout, token_ids, tokenizer), layout.insert_after, widths, False),
                              device=hidden.device, dtype=torch.bool)
        if bool(active.any()):
            hidden = hidden.clone()
            hidden[:, active] = model.local_memory.weaver.read_decoder_hidden(hidden[:, active], reader_memory)
    logits = model.logits(hidden.index_select(1, torch.tensor(predictors, device=hidden.device)))
    logits = logits + _end_mask(logits, token_ids)
    blocked = splice(_support_mask(layout, token_ids, tokenizer, max_calls), layout.insert_after, widths, False)
    blocked = [blocked[p] for p in predictors]
    if any(blocked):
        logits = logits.clone()
        rows = torch.tensor(blocked, dtype=torch.bool, device=logits.device)
        for request in token_ids.request_ids:
            logits[:, rows, request] = -torch.inf
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    return logp.gather(1, torch.tensor(targets, device=logp.device)[:, None]).squeeze(1)


def _end_mask(logits, token_ids):
    mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    mask.index_fill_(0, torch.tensor(token_ids.end_ids, device=logits.device), float("-inf"))
    return mask
