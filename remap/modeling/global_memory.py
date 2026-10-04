"""Global memory: a cached, question-conditioned summary of all input images.

Layout of the returned block (``N`` images, ``d`` hidden size)::

    N = 1      I1 (8 tokens)               + REL (8)   -> 16 tokens
    N = 2      I1, I2 (4 each)             + REL (8)   -> 16 tokens
    N = 3      I1..I3 (4 each)             + REL (4)   -> 16 tokens
    N = 4..10  I1..IN (4 each)             + REL (4)   -> 4N + 4 tokens

Each image span cross-attends only to its own image's visual tokens; the REL span
attends to all images.  Every span's queries are shifted by a *clause* vector:

* N <= 3: clauses come from the fact writer (:mod:`remap.modeling.fact_writer`),
  which writes a short structured description ``I1{..}; I2{..}; REL{..}.`` and
  averages its hidden states over each slot.  The block is computed once at prefill
  and reused for every Global request.
* N >= 4: clauses come from :class:`ClauseWriter`, a light module over per-image
  summaries and the question state.  The raw block is cached, and every request
  re-reads it with :class:`PrefixReadout` conditioned on the state that issued the
  request.

The raw block is type-stamped and passed through a residual bridge before insertion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

MAX_IMAGES = 10
CLAUSE_SLOTS_HIGH_N = 11      # I1..I10 then REL
REL_CLAUSE_HIGH_N = 10
REL_CLAUSE_LOW_N = 3          # I1..I3 then REL


@dataclass(frozen=True)
class Span:
    role: str
    start: int
    end: int
    image_index: Optional[int]

    @property
    def length(self) -> int:
        return self.end - self.start


def partition(image_count: int) -> Tuple[Span, ...]:
    if not 1 <= image_count <= MAX_IMAGES:
        raise ValueError("Global memory supports 1 to 10 images")
    image_width = 8 if image_count == 1 else 4
    relation_width = 8 if image_count <= 2 else 4
    spans = [Span(f"I{i + 1}", i * image_width, (i + 1) * image_width, i) for i in range(image_count)]
    offset = image_width * image_count
    return tuple(spans + [Span("REL", offset, offset + relation_width, None)])


def image_token_count(image_count: int) -> int:
    """Number of leading tokens typed as per-image (the rest are typed as relation)."""
    return 8 if image_count in (1, 2) else 12 if image_count == 3 else 4 * image_count


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention and pre-LN GELU FFN, both residual."""

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_ratio: float = 4.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.ln_ffn = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, inner), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(inner, hidden_size))

    def forward(self, query, kv, key_padding_mask=None, attn_mask=None):
        kv = self.ln_kv(kv)
        attended, _ = self.attn(self.ln_q(query), kv, kv, key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask, need_weights=False)
        query = query + attended
        return query + self.ffn(self.ln_ffn(query))


class GlobalWeaver(nn.Module):
    """Image and Relation Weaver: clause-shifted learnable queries over visual tokens."""

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.image_queries = nn.Parameter(torch.empty(1, 8, hidden_size))
        self.relation_queries = nn.Parameter(torch.empty(1, 8, hidden_size))
        self.image_embed = nn.Embedding(MAX_IMAGES, hidden_size)
        self.clause_norm = nn.LayerNorm(hidden_size)
        self.clause_projection = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList(CrossAttentionBlock(hidden_size, num_heads) for _ in range(num_layers))
        self.out_ln = nn.LayerNorm(hidden_size)
        for value in (self.image_queries, self.relation_queries, self.image_embed.weight):
            nn.init.normal_(value, std=0.02)

    def forward(self, visual_tokens: torch.Tensor, image_ids: torch.Tensor,
                clause_queries: torch.Tensor, image_count: int) -> torch.Tensor:
        """``visual_tokens`` [V, d], ``image_ids`` [V], ``clause_queries`` [4 or 11, d] -> [W, d]."""
        spans = partition(image_count)
        kv = visual_tokens.detach().float().unsqueeze(0)
        clauses = self.clause_projection(self.clause_norm(clause_queries.float()))
        blocked = torch.zeros(spans[-1].end, len(visual_tokens), dtype=torch.bool,
                              device=self.image_queries.device)
        queries = []
        for span in spans:
            if span.image_index is None:
                clause = REL_CLAUSE_LOW_N if image_count <= 3 else REL_CLAUSE_HIGH_N
                query = self.relation_queries[:, :span.length]
            else:
                clause = span.image_index
                query = self.image_queries[:, :span.length] + self.image_embed.weight[span.image_index]
                blocked[span.start:span.end] = image_ids != span.image_index
            queries.append(query + clauses[clause])
        query = torch.cat(queries, dim=1)
        mask = blocked.unsqueeze(0).expand(self.num_heads, -1, -1)
        for layer in self.layers:
            query = layer(query, kv, None, attn_mask=mask)
        return self.out_ln(query)[0].to(visual_tokens.dtype)


class ClauseWriter(nn.Module):
    """Clause queries for 4..10 images from per-image summaries and the question state."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.rank = rank
        self.question_norm = nn.LayerNorm(hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
        self.question_proj = nn.Linear(hidden_size, rank, bias=False)
        self.image_proj = nn.Linear(hidden_size, rank, bias=False)
        self.slot_embed = nn.Embedding(MAX_IMAGES, rank)
        self.image_out = nn.Linear(rank, hidden_size, bias=False)
        self.relation_out = nn.Linear(3 * rank, hidden_size, bias=False)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

    def forward(self, image_summaries: torch.Tensor, question_state: torch.Tensor) -> torch.Tensor:
        """``image_summaries`` [N, d], ``question_state`` [d] -> clause queries [11, d]."""
        summaries = image_summaries.unsqueeze(0) if image_summaries.ndim == 2 else image_summaries
        question = question_state.unsqueeze(0) if question_state.ndim == 1 else question_state
        count = summaries.shape[1]
        dtype = self.question_proj.weight.dtype
        question_rank = self.question_proj(self.question_norm(question.detach().to(dtype=dtype)))
        image_rank = self.image_proj(self.image_norm(summaries.detach().to(dtype=dtype)))
        slots = self.slot_embed(torch.arange(count, device=summaries.device)).unsqueeze(0)
        conditioned = image_rank + question_rank.unsqueeze(1) + slots
        image_clauses = self.image_out(F.silu(conditioned))
        weights = torch.arange(1, count + 1, device=summaries.device, dtype=dtype)
        weights = weights / weights.sum()
        ordered = torch.sum(weights.reshape(1, count, 1) * conditioned, dim=1)
        logits = torch.sum(conditioned * question_rank.unsqueeze(1), dim=-1) / math.sqrt(float(self.rank))
        attended = torch.sum(torch.softmax(logits, dim=1).unsqueeze(-1) * conditioned, dim=1)
        relation = self.relation_out(F.silu(torch.cat((question_rank, ordered, attended), dim=-1)))
        clauses = torch.zeros(question.shape[0], CLAUSE_SLOTS_HIGH_N, image_clauses.shape[-1],
                              device=question.device, dtype=image_clauses.dtype)
        clauses[:, :count] = image_clauses
        clauses[:, REL_CLAUSE_HIGH_N] = relation
        return clauses[0]


class ResidualBridge(nn.Module):
    """Mean-centred low-rank residual that maps raw memory into the decoder's input space."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.neutral_shrink_logit = nn.Parameter(torch.zeros(()))
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """``memory`` [1, W, d] -> [1, W, d] (same dtype)."""
        real = memory.float()
        neutral = real.mean(dim=1, keepdim=True).expand_as(real)
        mapped = self.up(F.silu(self.down(self.norm(real - neutral))))
        gain = 2.0 * torch.sigmoid(self.gate_logit)
        shrink = torch.tanh(self.neutral_shrink_logit)
        return (real + shrink * (neutral - real) + gain * mapped).to(dtype=memory.dtype)


class PrefixReadout(nn.Module):
    """Re-read a cached block conditioned on the decoder state that issued the request."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_down = nn.Linear(hidden_size, rank, bias=False)
        self.query_down = nn.Linear(hidden_size, rank, bias=False)
        self.output_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.memory_down.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.query_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.output_up.weight)

    def forward(self, memory: torch.Tensor, request_state: torch.Tensor) -> torch.Tensor:
        """``memory`` [1, W, d], ``request_state`` [1, d] -> [1, W, d]."""
        query = request_state.unsqueeze(0) if request_state.ndim == 1 else request_state
        fused = F.silu(self.memory_down(self.memory_norm(memory.float()))
                       + self.query_down(self.query_norm(query.detach().float()).unsqueeze(1)))
        return (memory.float() + self.output_up(fused)).to(dtype=memory.dtype)


class GlobalMemory(nn.Module):
    def __init__(self, hidden_size: int, *, rank: int = 64, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.weaver = GlobalWeaver(hidden_size, num_layers, num_heads)
        self.clause_writer = ClauseWriter(hidden_size, rank)
        self.bridge_low_n = ResidualBridge(hidden_size, rank)
        self.bridge_high_n = ResidualBridge(hidden_size, rank)
        self.readout = PrefixReadout(hidden_size, rank)

    def raw_low_n(self, visual_tokens, image_ids, clause_queries, image_count) -> torch.Tensor:
        """Raw block for 1..3 images from fact-writer clauses ([4, d])."""
        return self.weaver(visual_tokens, image_ids, clause_queries.detach(), image_count)

    def raw_high_n(self, visual_tokens, image_ids, image_summaries, question_state, image_count):
        """Raw block for 4..10 images; clauses come from :class:`ClauseWriter`."""
        clauses = self.clause_writer(image_summaries.detach(), question_state.detach())
        return self.weaver(visual_tokens, image_ids, clauses, image_count)

    def bridge(self, raw: torch.Tensor, image_count: int, type_embedding) -> torch.Tensor:
        """Type-stamp ([W, d] -> [1, W, d]) and bridge the raw block."""
        typed = type_embedding.stamp_global(raw.unsqueeze(0), image_token_count(image_count))
        bridge = self.bridge_low_n if image_count <= 3 else self.bridge_high_n
        return bridge(typed)
