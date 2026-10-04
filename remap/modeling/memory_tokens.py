"""Memory special tokens and their trainable embedding / output rows.

ReMAP adds four special tokens to the Qwen2.5-VL vocabulary:

* ``<mem_g>`` and ``<mem_l>`` are *request* tokens.  The policy emits them to ask
  for Global or Local memory.
* ``<mem_g_end>`` and ``<mem_l_end>`` are *end* markers.  They are written by the
  environment after the inserted latent block and are never sampled.

The large embedding and LM-head matrices stay frozen.  Their rows for these tokens
are replaced by a small set of trainable rows (:class:`MemoryRows`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

MEM_G = "<mem_g>"
MEM_L = "<mem_l>"
MEM_G_END = "<mem_g_end>"
MEM_L_END = "<mem_l_end>"
MEMORY_TOKENS = (MEM_G, MEM_L, MEM_G_END, MEM_L_END)


@dataclass(frozen=True)
class MemoryTokenIds:
    mem_g: int
    mem_l: int
    mem_g_end: int
    mem_l_end: int

    @property
    def request_ids(self) -> List[int]:
        return [self.mem_g, self.mem_l]

    @property
    def end_ids(self) -> List[int]:
        return [self.mem_g_end, self.mem_l_end]

    @property
    def all_ids(self) -> List[int]:
        return [self.mem_g, self.mem_l, self.mem_g_end, self.mem_l_end]

    def kind(self, token_id: int) -> Optional[str]:
        """Return ``'global'`` / ``'local'`` for a request token, else ``None``."""
        if token_id == self.mem_g:
            return "global"
        if token_id == self.mem_l:
            return "local"
        return None

    def end_for(self, request_id: int) -> int:
        if request_id == self.mem_g:
            return self.mem_g_end
        if request_id == self.mem_l:
            return self.mem_l_end
        raise ValueError(f"{request_id} is not a memory request token")


def register_memory_tokens(tokenizer, model=None) -> MemoryTokenIds:
    """Add the four tokens to ``tokenizer`` (and size ``model``'s vocabulary for them).

    New rows of the frozen embedding and LM head are initialised to the mean of the
    existing vocabulary.  These rows are never read directly: :class:`MemoryRows`
    overrides them in :meth:`ReMAPForConditionalGeneration.embed` and ``logits``.
    """
    missing = [t for t in MEMORY_TOKENS
               if tokenizer.convert_tokens_to_ids(t) in (None, tokenizer.unk_token_id)]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing},
                                     replace_additional_special_tokens=False)
    ids = MemoryTokenIds(*(tokenizer.convert_tokens_to_ids(t) for t in MEMORY_TOKENS))
    if model is not None:
        old_rows = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(max(len(tokenizer), old_rows))
        model.config.vocab_size = model.get_input_embeddings().weight.shape[0]
        new_ids = [tokenizer.convert_tokens_to_ids(t) for t in missing]
        if new_ids:
            real = len(tokenizer) - len(missing)
            with torch.no_grad():
                for weight in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
                    mean = weight[:real].mean(dim=0, keepdim=True).to(weight.dtype)
                    for index in new_ids:
                        weight[index].copy_(mean[0])
    return ids


class MemoryRows(nn.Module):
    """Trainable input rows for all four tokens and output rows for the two requests."""

    def __init__(self, ids: MemoryTokenIds, hidden_size: int):
        super().__init__()
        self.ids = ids
        self.embed_request = nn.Parameter(torch.zeros(2, hidden_size))
        self.embed_end = nn.Parameter(torch.zeros(2, hidden_size))
        self.head_request = nn.Parameter(torch.zeros(2, hidden_size))
        self.register_buffer("request_ids", torch.tensor(ids.request_ids), persistent=False)

    @torch.no_grad()
    def init_from_model(self, model) -> "MemoryRows":
        embed = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        self.embed_request.copy_(embed[self.ids.request_ids])
        self.embed_end.copy_(embed[self.ids.end_ids])
        self.head_request.copy_(head[self.ids.request_ids])
        return self

    def overlay_embeddings(self, embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        for row, token in enumerate(self.ids.request_ids):
            mask = (input_ids == token).unsqueeze(-1)
            embeds = torch.where(mask, self.embed_request[row].to(embeds.dtype), embeds)
        for row, token in enumerate(self.ids.end_ids):
            mask = (input_ids == token).unsqueeze(-1)
            embeds = torch.where(mask, self.embed_end[row].to(embeds.dtype), embeds)
        return embeds

    def overlay_logits(self, logits: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        request = hidden @ self.head_request.t().to(hidden.dtype)
        return logits.index_copy(-1, self.request_ids.to(logits.device), request.to(logits.dtype))


def forbidden_response_ids(tokenizer, ids: MemoryTokenIds) -> List[int]:
    """Every special token except EOS and the four memory tokens is banned from responses."""
    special = {int(v) for v in tokenizer.all_special_ids}
    eos = int(tokenizer.eos_token_id)
    return sorted(special - set(ids.all_ids) - {eos})


def logit_mask(vocab_size: int, banned: Sequence[int], device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive mask with ``-inf`` at ``banned`` positions."""
    mask = torch.zeros(vocab_size, device=device, dtype=dtype)
    if len(banned):
        mask.index_fill_(0, torch.as_tensor(list(banned), device=device, dtype=torch.long), float("-inf"))
    return mask
