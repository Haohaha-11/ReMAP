"""Local memory: region-level evidence selected and re-encoded for the current state.

On a ``<mem_l>`` request (see :meth:`remap.modeling.ReMAPForConditionalGeneration.local_block`):

1. :class:`SourceSelector` keeps all images when N <= 3, otherwise scores the images
   against the request state and keeps the top images until their probability mass
   reaches 0.85 (at most three).
2. For each kept image, Twig scores its visual tokens and
   :func:`remap.modeling.twig.relative_peak_bbox` turns the map into a box; the
   region is cropped (:func:`crop_region`), re-encoded at higher resolution by the
   frozen vision encoder, and compressed by :class:`LocalWeaver` into 32 tokens.
3. :class:`GlobalConditioner` shifts the request state with the cached relation memory
   before the weaver reads it, and mixes the per-image blocks slot by slot.
4. Per-image blocks are packed into one 32-token budget (32 / 16+16 / 12+10+10,
   strongest image first, :func:`pack_k32`) and fused with summaries of *all* images
   by :class:`SetContext`.

The weaver also returns a *crop residual*; it is kept as reader memory and, after a
Local request, adapts the hidden state that predicts the final answer
(:meth:`LocalWeaver.read_decoder_hidden`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

LOCAL_TOKENS = 32
PARTITIONS = {1: (32,), 2: (16, 16), 3: (12, 10, 10)}
MAX_IMAGES = 10
MAX_SELECTED = 3


# --------------------------------------------------------------------------- cropping
def crop_region(image, bbox, margin: float = 0.1, min_frac: float = 0.12):
    """Expand ``bbox`` by ``margin`` of its size per side, enforce a minimum size, crop."""
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= margin * box_w
    x2 += margin * box_w
    y1 -= margin * box_h
    y2 += margin * box_h
    crop_w, crop_h = x2 - x1, y2 - y1
    min_w, min_h = min_frac * width, min_frac * height
    if crop_w < min_w:
        cx = (x1 + x2) / 2
        x1, x2 = cx - min_w / 2, cx + min_w / 2
    if crop_h < min_h:
        cy = (y1 + y2) / 2
        y1, y2 = cy - min_h / 2, cy + min_h / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    crop = image.crop((x1, y1, x2, y2))
    w, h = crop.size
    if w < 28 or h < 28:
        crop = crop.resize((max(w, 28), max(h, 28)))
    return crop


# --------------------------------------------------------------------------- weaver
class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_ratio: float = 4.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, inner), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(inner, hidden_size))

    def forward(self, query, kv, key_padding_mask=None):
        attended, _ = self.attn(self.ln_q(query), self.ln_kv(kv), self.ln_kv(kv),
                                key_padding_mask=key_padding_mask, need_weights=False)
        query = query + attended
        return query + self.ffn(self.ln_ffn(query))


def _context_code(index: int, hidden: int, device, dtype) -> torch.Tensor:
    axis = torch.arange(hidden, device=device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * (2 * torch.div(axis, 2, rounding_mode="floor")) / max(1, hidden))
    phase = float(index + 1) * frequency
    code = torch.where((axis.long() % 2) == 0, torch.sin(phase), torch.cos(phase))
    return (0.02 * code).to(dtype)


class LocalWeaver(nn.Module):
    """Learnable queries cross-attending to region tokens, conditioned on state and box."""

    def __init__(self, hidden_size: int, *, num_tokens: int = LOCAL_TOKENS, num_layers: int = 2,
                 num_heads: int = 8, context_rank: int = 64, max_context_images: int = 64):
        super().__init__()
        d, r = hidden_size, context_rank
        self.hidden_size = d
        self.num_tokens = num_tokens
        self.local_queries = nn.Parameter(torch.randn(1, num_tokens, d) * 0.02)
        self.query_cond_proj = nn.Linear(d, d)
        self.state_kv_proj = nn.Linear(d, d)
        self.bbox_proj = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(_CrossAttentionBlock(d, num_heads) for _ in range(num_layers))
        self.out_ln = nn.LayerNorm(d)
        # All-image context (image summaries and anchor-relative relations).
        self.context_summary_norm = nn.LayerNorm(d)
        self.context_relation_norm = nn.LayerNorm(d)
        self.context_summary_adapter = nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Linear(r, d))
        self.context_relation_adapter = nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Linear(r, d))
        self.context_slot_embed = nn.Embedding(max_context_images, d)
        self.context_role_embed = nn.Embedding(4, d)
        # Low-rank residual reads of the context and of the crop itself.
        self.context_residual_norm = nn.LayerNorm(d)
        self.context_residual_q = nn.Linear(d, r, bias=False)
        self.context_residual_k = nn.Linear(d, r, bias=False)
        self.context_residual_v = nn.Linear(d, r, bias=False)
        self.context_residual_out = nn.Linear(r, d, bias=False)
        self.context_crop_norm = nn.LayerNorm(d)
        self.context_crop_q = nn.Linear(d, r, bias=False)
        self.context_crop_k = nn.Linear(d, r, bias=False)
        self.context_crop_v = nn.Linear(d, r, bias=False)
        self.context_crop_out = nn.Linear(r, d, bias=False)
        # Answer-time readers.
        self.reader_hidden_norm = nn.LayerNorm(d)
        self.reader_hidden_in = nn.Linear(d, r, bias=False)
        self.reader_hidden_out = nn.Linear(r, d, bias=False)
        self.reader_memory_norm = nn.LayerNorm(d)
        self.reader_memory_q = nn.Linear(d, r, bias=False)
        self.reader_memory_k = nn.Linear(d, r, bias=False)
        self.reader_memory_v = nn.Linear(d, r, bias=False)
        self.reader_memory_out = nn.Linear(r, d, bias=False)
        self.crop_scale = 1.0
        self.reader_hidden_scale = 1.0
        self.reader_memory_scale = 1.0
        for module in (self.context_summary_adapter[-1], self.context_relation_adapter[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for module in (self.context_residual_out, self.context_crop_out, self.reader_hidden_out,
                       self.reader_memory_out):
            nn.init.zeros_(module.weight)
        nn.init.normal_(self.context_slot_embed.weight, std=0.02)
        nn.init.normal_(self.context_role_embed.weight, std=0.02)

    def image_context(self, image_states: Sequence[torch.Tensor], *, anchor_index: int,
                      roi_image_index: int) -> torch.Tensor:
        """Per-image summaries (with slot and role embeddings) then anchor-relative relations."""
        means = [(s if s.ndim == 3 else s.unsqueeze(0)).mean(dim=1) for s in image_states]
        mean = torch.stack(means, dim=1)
        count, device, dtype = len(image_states), mean.device, mean.dtype
        adapter_dtype = self.context_summary_adapter[0].weight.dtype
        summary = mean + self.context_summary_adapter(self.context_summary_norm(mean).to(adapter_dtype)).to(dtype)
        summary = summary + self.context_slot_embed(torch.arange(count, device=device)).to(dtype).unsqueeze(0)
        summary = summary + self.context_role_embed(
            torch.zeros(count, device=device, dtype=torch.long)).to(dtype).unsqueeze(0)
        anchor_mask = torch.zeros((1, count, 1), device=device, dtype=dtype)
        roi_mask = torch.zeros((1, count, 1), device=device, dtype=dtype)
        anchor_mask[:, anchor_index] = 1
        roi_mask[:, roi_image_index] = 1
        summary = summary + anchor_mask * self.context_role_embed(torch.tensor(2, device=device)).to(dtype).view(1, 1, -1)
        summary = summary + roi_mask * self.context_role_embed(torch.tensor(3, device=device)).to(dtype).view(1, 1, -1)
        relations = []
        anchor = mean[:, anchor_index]
        relation_dtype = self.context_relation_adapter[0].weight.dtype
        for index in range(count):
            if index == anchor_index:
                continue
            delta = mean[:, index] - anchor
            token = delta + self.context_relation_adapter(self.context_relation_norm(delta).to(relation_dtype)).to(dtype)
            token = token + self.context_slot_embed(torch.tensor(index, device=device)).to(dtype)
            token = token + self.context_role_embed(torch.tensor(1, device=device)).to(dtype)
            relations.append(token.unsqueeze(1))
        return torch.cat([summary] + relations, dim=1) if relations else summary

    @staticmethod
    def _low_rank_read(query, keys_source, norm, q_proj, k_proj, v_proj):
        normed = norm(keys_source.to(norm.weight.dtype)).to(q_proj.weight.dtype)
        q = q_proj(query.to(q_proj.weight.dtype))
        k, v = k_proj(normed), v_proj(normed)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(1, q.shape[-1]))
        return torch.matmul(torch.softmax(scores.float(), dim=-1).to(v.dtype), v)

    def forward(self, region_tokens: torch.Tensor, *, query_state: torch.Tensor, bbox: torch.Tensor,
                image_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(memory, crop_residual)``, both [B, num_tokens, d]."""
        x = region_tokens
        batch = x.shape[0]
        bbox_vec = self.bbox_proj(bbox.to(x.dtype))
        state = query_state if query_state.dim() == 2 else query_state.mean(dim=1)
        kv = torch.cat([x, self.state_kv_proj(state).unsqueeze(1), bbox_vec.unsqueeze(1)], dim=1)
        query = self.local_queries.expand(batch, -1, -1).to(x.dtype)
        query = query + self.query_cond_proj(state).unsqueeze(1) + bbox_vec.unsqueeze(1)
        for layer in self.layers:
            query = layer(query, kv, None)
        base = self.out_ln(query)
        result = base
        if image_context is not None:
            read = self._low_rank_read(result, image_context.to(device=x.device, dtype=x.dtype),
                                       self.context_residual_norm, self.context_residual_q,
                                       self.context_residual_k, self.context_residual_v)
            result = result + self.context_residual_out(read).to(base.dtype)
        crop_read = self._low_rank_read(result, x, self.context_crop_norm, self.context_crop_q,
                                        self.context_crop_k, self.context_crop_v)
        crop_residual = self.context_crop_out(crop_read).to(base.dtype)
        result = result + crop_residual * float(self.crop_scale)
        return result, crop_residual

    def read_decoder_hidden(self, hidden: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        """Adapt answer-time hidden states with the last Local request's reader memory."""
        delta = torch.zeros_like(hidden)
        normed = self.reader_hidden_norm(hidden).to(self.reader_hidden_in.weight.dtype)
        hidden_delta = self.reader_hidden_out(F.gelu(self.reader_hidden_in(normed))).to(hidden.dtype)
        delta = delta + hidden_delta * float(self.reader_hidden_scale)
        norm_dtype = self.reader_memory_norm.weight.dtype
        q = self.reader_memory_q(self.reader_memory_norm(hidden.to(norm_dtype)).to(self.reader_memory_q.weight.dtype))
        memory = self.reader_memory_norm(memory_tokens.to(norm_dtype)).to(self.reader_memory_k.weight.dtype)
        k, v = self.reader_memory_k(memory), self.reader_memory_v(memory)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(1, q.shape[-1]))
        attended = torch.matmul(torch.softmax(scores.float(), dim=-1).to(v.dtype), v)
        delta = delta + self.reader_memory_out(attended).to(hidden.dtype) * float(self.reader_memory_scale)
        return hidden + delta


# --------------------------------------------------------------------------- selection and fusion
@dataclass
class Selection:
    positions: Tuple[int, ...]          # selected image indices, strongest first
    scores: Optional[torch.Tensor]      # None when every image is kept (N <= 3)


class SourceSelector(nn.Module):
    def __init__(self, hidden_size: int, *, rank: int = 64, max_selected: int = MAX_SELECTED,
                 mass_threshold: float = 0.85):
        super().__init__()
        self.max_selected = max_selected
        self.mass_threshold = mass_threshold
        self.query_norm = nn.LayerNorm(hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, rank, bias=False)
        self.image_proj = nn.Linear(hidden_size, rank, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, query_state: torch.Tensor, image_states: Sequence[torch.Tensor]) -> Selection:
        count = len(image_states)
        if count <= 3:
            return Selection(tuple(range(count)), None)
        dtype = self.query_proj.weight.dtype
        query = F.normalize(self.query_proj(self.query_norm(query_state.detach().to(dtype))).float(), dim=-1)
        summaries = torch.stack([s.detach().mean(dim=1) for s in image_states], dim=1)
        images = F.normalize(self.image_proj(self.image_norm(summaries.to(dtype))).float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        scores = torch.einsum("br,bnr->bn", query, images)[0] * scale
        probabilities = torch.softmax(scores, dim=0).detach().cpu().tolist()
        values = scores.detach().cpu().tolist()
        ranked = sorted(range(count), key=lambda i: (-values[i], i))
        selected, mass = [], 0.0
        for position in ranked[: self.max_selected]:
            selected.append(position)
            mass += float(probabilities[position])
            if mass >= self.mass_threshold:
                break
        return Selection(tuple(selected), scores)


class GlobalConditioner(nn.Module):
    """Condition the request state on relation memory; mix per-image blocks slot-wise."""

    def __init__(self, hidden_size: int, *, rank: int = 64, latent_tokens: int = LOCAL_TOKENS):
        super().__init__()
        self.rank = rank
        self.state_norm = nn.LayerNorm(hidden_size)
        self.global_norm = nn.LayerNorm(hidden_size)
        self.state_q = nn.Linear(hidden_size, rank, bias=False)
        self.global_k = nn.Linear(hidden_size, rank, bias=False)
        self.global_v = nn.Linear(hidden_size, rank, bias=False)
        self.query_out = nn.Linear(rank, hidden_size, bias=False)
        self.image_norm = nn.LayerNorm(hidden_size)
        self.allocation_q = nn.Linear(hidden_size, rank, bias=False)
        self.allocation_k = nn.Linear(hidden_size, rank, bias=False)
        self.slot_queries = nn.Parameter(torch.randn(latent_tokens, rank) * 0.02)
        self.slot_gate = nn.Parameter(torch.zeros(latent_tokens))
        nn.init.zeros_(self.query_out.weight)

    def condition(self, state: torch.Tensor, relation_tokens: torch.Tensor) -> torch.Tensor:
        q = self.state_q(self.state_norm(state.to(self.state_norm.weight.dtype)).to(self.state_q.weight.dtype)).unsqueeze(1)
        normed = self.global_norm(relation_tokens.to(self.global_norm.weight.dtype)).to(self.global_k.weight.dtype)
        k, v = self.global_k(normed), self.global_v(normed)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(1, self.rank))
        attended = torch.matmul(torch.softmax(scores.float(), dim=-1).to(v.dtype), v)[:, 0]
        return state + self.query_out(attended).to(state.dtype)

    def mix(self, blocks: Sequence[torch.Tensor], image_states: Sequence[torch.Tensor],
            state: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        summary = torch.stack([s.detach().mean(dim=1) for s in image_states], dim=1)
        query = self.allocation_q(self.state_norm(state.to(self.state_norm.weight.dtype)).to(self.allocation_q.weight.dtype))
        images = self.allocation_k(self.image_norm(summary.to(self.image_norm.weight.dtype)).to(self.allocation_k.weight.dtype))
        slots = query.unsqueeze(1) + self.slot_queries.to(query.dtype).unsqueeze(0)
        logits = torch.einsum("bkr,bir->bki", slots, images) / math.sqrt(max(1, self.rank))
        weights = torch.softmax(logits.float(), dim=-1).to(blocks[0].dtype)
        adaptive = torch.einsum("bki,bikd->bkd", weights, torch.stack(list(blocks), dim=1))
        gate = torch.tanh(self.slot_gate).to(adaptive.dtype).view(1, -1, 1)
        return packed + gate * (adaptive - packed)


class SetContext(nn.Module):
    """Fuse a 16-token memory chunk with summaries of every input image."""

    def __init__(self, hidden_size: int, *, rank: int = 64, chunk_tokens: int = 16):
        super().__init__()
        self.rank = rank
        self.chunk_tokens = chunk_tokens
        self.query_norm = nn.LayerNorm(hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, rank, bias=False)
        self.image_key = nn.Linear(hidden_size, rank, bias=False)
        self.image_value = nn.Linear(hidden_size, rank, bias=False)
        self.slot_embed = nn.Embedding(MAX_IMAGES, rank)
        self.count_embed = nn.Embedding(MAX_IMAGES + 1, rank)
        self.out_proj = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.slot_embed.weight, std=0.02)
        nn.init.zeros_(self.count_embed.weight)
        nn.init.zeros_(self.out_proj.weight)

    def _fuse_chunk(self, chunk: torch.Tensor, image_states: Sequence[torch.Tensor]) -> torch.Tensor:
        count = len(image_states)
        dtype = self.query_proj.weight.dtype
        summaries = torch.stack([s.detach().mean(dim=1) for s in image_states], dim=1).to(dtype=dtype)
        images = self.image_norm(summaries)
        stamps = self.slot_embed(torch.arange(count, device=chunk.device)).unsqueeze(0)
        stamps = stamps + self.count_embed(torch.tensor(count, device=chunk.device)).reshape(1, 1, -1)
        keys = self.image_key(images) + stamps
        values = self.image_value(images) + stamps
        query = self.query_proj(self.query_norm(chunk.detach().to(dtype=dtype)))
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(float(max(1, self.rank)))
        weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return chunk + self.out_proj(torch.matmul(weights, values)).to(chunk.dtype)

    def forward(self, memory: torch.Tensor, image_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """``memory`` [1, 32, d]: each 16-token half is fused independently."""
        chunks = memory.split(self.chunk_tokens, dim=1)
        return torch.cat([self._fuse_chunk(chunk, image_states) for chunk in chunks], dim=1)


def pack_k32(blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Take slices of the per-image blocks (already ordered strongest first) into 32 tokens."""
    if len(blocks) == 1:
        return blocks[0]
    parts, offset = [], 0
    for block, width in zip(blocks, PARTITIONS[len(blocks)]):
        parts.append(block[:, offset:offset + width])
        offset += width
    return torch.cat(parts, dim=1)


class LocalMemory(nn.Module):
    def __init__(self, hidden_size: int, *, selector_rank: int = 64, context_rank: int = 64,
                 conditioner_rank: int = 64, set_context_rank: int = 64):
        super().__init__()
        self.weaver = LocalWeaver(hidden_size, context_rank=context_rank)
        self.selector = SourceSelector(hidden_size, rank=selector_rank)
        self.conditioner = GlobalConditioner(hidden_size, rank=conditioner_rank)
        self.set_context = SetContext(hidden_size, rank=set_context_rank)
