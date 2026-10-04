"""Twig: the region locator used by Local memory.

Twig is a short side branch of the decoder (three decoder layers, initialised from
Q-Zoom's SD-RPN branch) that starts from the hidden states of decoder layer
``branch_layer``.  For a query position it scores every visual token of an image
with the scaled dot product between the last Twig layer's query and key
projections; the score map is turned into a box by :func:`relative_peak_bbox`.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


class TwigLocator(nn.Module):
    def __init__(self, text_config, *, num_layers: int = 3, branch_layer: int = 18):
        super().__init__()
        self.branch_layer = int(branch_layer)
        self.layers = nn.ModuleList(
            Qwen2_5_VLDecoderLayer(text_config, layer_idx=self.branch_layer + index)
            for index in range(num_layers))

    def _query_key(self, decoder, hidden, position_ids, attention_mask):
        length = hidden.shape[1]
        cache_position = torch.arange(length, device=hidden.device)
        position_embeddings = decoder.rotary_emb(hidden, position_ids)
        causal_mask = decoder._update_causal_mask(attention_mask, hidden, cache_position, None, False)
        states = hidden
        for layer in self.layers[:-1]:
            states = layer(states, attention_mask=causal_mask, position_ids=position_ids,
                           past_key_value=None, use_cache=False, cache_position=cache_position,
                           position_embeddings=position_embeddings)[0]
        attention = self.layers[-1].self_attn
        normed = self.layers[-1].input_layernorm(states)
        query, key = attention.q_proj(normed), attention.k_proj(normed)
        heads, head_dim, kv_heads = attention.num_heads, attention.head_dim, attention.num_key_value_heads
        query = query.view(1, length, heads, head_dim).transpose(1, 2)
        key = key.view(1, length, kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, attention.rope_scaling["mrope_section"])
        return query, repeat_kv(key, heads // kv_heads), head_dim

    @staticmethod
    def _score_map(query, key, head_dim, index, visual_indices, grid_hw):
        scores = torch.matmul(query[:, :, int(index):int(index) + 1, :], key[:, :, visual_indices, :].transpose(-2, -1))
        rows, cols = grid_hw
        return (scores / math.sqrt(head_dim)).mean(dim=1)[0, 0].view(rows, cols)

    def heat(self, decoder, hidden: torch.Tensor, position_ids: torch.Tensor,
             query_indices: Sequence[int], visual_indices: torch.Tensor,
             grid_hw: Tuple[int, int], attention_mask=None) -> List[torch.Tensor]:
        """Score map over one image's visual tokens for each query position.

        ``hidden`` is ``hidden_states[branch_layer]`` of a full (cache-free) decoder
        pass and ``decoder`` is the base ``Qwen2_5_VLModel`` (its rotary embedding
        and causal-mask helper are reused).
        """
        query, key, head_dim = self._query_key(decoder, hidden, position_ids, attention_mask)
        return [self._score_map(query, key, head_dim, index, visual_indices, grid_hw) for index in query_indices]

    def image_heats(self, decoder, hidden, position_ids, query_index: int, images, attention_mask=None):
        """Score maps of one query over several images; ``images`` is [(visual_indices, grid_hw)]."""
        query, key, head_dim = self._query_key(decoder, hidden, position_ids, attention_mask)
        return [self._score_map(query, key, head_dim, query_index, indices, grid) for indices, grid in images]


def relative_peak_bbox(heat: torch.Tensor, width: int, height: int, *,
                       threshold: float = 0.5, pad: float = 0.06) -> List[float]:
    """Box (in pixels) around grid cells whose softmax mass is >= ``threshold`` x peak."""
    rows, cols = heat.shape
    probs = torch.softmax(heat.flatten().float(), dim=-1).view(rows, cols)
    ys, xs = torch.where(probs >= probs.max() * threshold)
    if ys.numel() == 0:
        flat = int(torch.argmax(probs))
        ys, xs = torch.tensor([flat // cols]), torch.tensor([flat % cols])
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    return [max(0.0, c0 / cols - pad) * width, max(0.0, r0 / rows - pad) * height,
            min(1.0, c1 / cols + pad) * width, min(1.0, r1 / rows + pad) * height]
