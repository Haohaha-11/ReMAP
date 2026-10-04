"""Relation memory: a frozen, answer-free summary that conditions Local requests.

The producer reads the prefill visual tokens of every image (with image/row/column
ids) and the text tokens of the prompt, forms 4 tokens per image, then 16 relation
tokens over those image tokens and the question.  It is computed once per prompt;
:class:`remap.modeling.local_memory.GlobalConditioner` reads it on every Local
request.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RelationConfig:
    hidden_size: int
    rank: int = 128
    num_heads: int = 8
    per_image_tokens: int = 4
    memory_tokens: int = 16
    per_image_layers: int = 1
    relation_layers: int = 2
    ffn_ratio: float = 4.0
    max_images: int = 32
    grid_size: int = 64


class _RankBlock(nn.Module):
    def __init__(self, cfg: RelationConfig):
        super().__init__()
        rank = cfg.rank
        self.q_norm = nn.LayerNorm(rank)
        self.kv_norm = nn.LayerNorm(rank)
        self.attention = nn.MultiheadAttention(rank, cfg.num_heads, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(rank)
        inner = int(rank * cfg.ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(rank, inner), nn.GELU(), nn.Linear(inner, rank))

    def forward(self, query, key_value, *, key_padding_mask=None, attention_mask=None):
        normed = self.kv_norm(key_value)
        attended, _ = self.attention(self.q_norm(query), normed, normed, key_padding_mask=key_padding_mask,
                                     attn_mask=attention_mask, need_weights=False)
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class RelationProducer(nn.Module):
    def __init__(self, cfg: RelationConfig):
        super().__init__()
        self.cfg = cfg
        d, rank = cfg.hidden_size, cfg.rank
        self.visual_norm = nn.LayerNorm(d)
        self.visual_down = nn.Linear(d, rank, bias=False)
        self.question_norm = nn.LayerNorm(d)
        self.question_down = nn.Linear(d, rank, bias=False)
        self.image_embedding = nn.Embedding(cfg.max_images, rank)
        self.row_embedding = nn.Embedding(cfg.grid_size, rank)
        self.col_embedding = nn.Embedding(cfg.grid_size, rank)
        self.visual_type = nn.Parameter(torch.zeros(1, 1, rank))
        self.question_type = nn.Parameter(torch.zeros(1, 1, rank))
        self.per_image_queries = nn.Parameter(torch.empty(1, cfg.per_image_tokens, rank))
        self.relation_queries = nn.Parameter(torch.empty(1, cfg.memory_tokens, rank))
        self.per_image_blocks = nn.ModuleList(_RankBlock(cfg) for _ in range(cfg.per_image_layers))
        self.relation_blocks = nn.ModuleList(_RankBlock(cfg) for _ in range(cfg.relation_layers))
        self.output_norm = nn.LayerNorm(rank)
        self.output_up = nn.Linear(rank, d, bias=False)
        for parameter in (self.per_image_queries, self.relation_queries, self.visual_type, self.question_type):
            nn.init.normal_(parameter, std=0.02)
        for embedding in (self.image_embedding, self.row_embedding, self.col_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(self.output_up.weight, std=0.002)

    def forward(self, visual_tokens, *, image_ids, row_ids, col_ids, question_tokens, num_images):
        """``visual_tokens`` [1, V, d] with ids [1, V]; ``question_tokens`` [1, T, d] -> [1, 16, d] (fp32)."""
        cfg = self.cfg
        visual = (self.visual_down(self.visual_norm(visual_tokens.float()))
                  + self.image_embedding(image_ids.clamp_max(cfg.max_images - 1))
                  + self.row_embedding(row_ids.clamp_max(cfg.grid_size - 1))
                  + self.col_embedding(col_ids.clamp_max(cfg.grid_size - 1))
                  + self.visual_type)
        question = self.question_down(self.question_norm(question_tokens.float())) + self.question_type
        question_summary = question.mean(dim=1, keepdim=True)
        groups = [self.per_image_queries + self.image_embedding.weight[i][None, None, :] + question_summary
                  for i in range(num_images)]
        per_image = torch.cat(groups, dim=1).expand(visual_tokens.shape[0], -1, -1)
        rows = [(image_ids != i)[:, None, :].expand(image_ids.shape[0], cfg.per_image_tokens, image_ids.shape[1])
                for i in range(num_images)]
        mask = torch.cat(rows, dim=1).repeat_interleave(cfg.num_heads, dim=0)
        for block in self.per_image_blocks:
            per_image = block(per_image, visual, key_padding_mask=None, attention_mask=mask)
        relation_kv = torch.cat([per_image, question], dim=1)
        relation = (self.relation_queries + question_summary).expand(visual_tokens.shape[0], -1, -1)
        for block in self.relation_blocks:
            relation = block(relation, relation_kv, key_padding_mask=None)
        return self.output_up(self.output_norm(relation))


def grid_ids(image_grid_thw: torch.Tensor, device):
    """Image index, row and column of every merged (2x2) visual token."""
    images, rows, cols = [], [], []
    for index, (t, h, w) in enumerate(image_grid_thw.tolist()):
        hh, ww = h // 2, w // 2
        count = t * hh * ww
        rows.append(torch.arange(hh, device=device).repeat_interleave(ww).repeat(t).long())
        cols.append(torch.arange(ww, device=device).repeat(hh * t).long())
        images.append(torch.full((count,), index, device=device, dtype=torch.long))
    return torch.cat(images), torch.cat(rows), torch.cat(cols)
