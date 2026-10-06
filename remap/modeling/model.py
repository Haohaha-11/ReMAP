"""ReMAP: Qwen2.5-VL with reasoning-time Global and Local latent visual memory."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn

from .global_memory import GlobalMemory
from .local_memory import LocalMemory, crop_region, pack_k32
from .memory_tokens import MemoryRows, MemoryTokenIds, forbidden_response_ids, register_memory_tokens
from .relation import RelationConfig, RelationProducer, grid_ids
from .twig import TwigLocator, relative_peak_bbox

GLOBAL, LOCAL, RELATION = 0, 1, 2
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class ReMAPConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    hidden_size: int = 3584
    # Decoder layer whose last-position state issues a request (Local query, Global readout).
    request_layer: int = 12
    # Twig locator.
    twig_layers: int = 3
    twig_branch_layer: int = 18
    twig_lora_r: int = 8
    twig_lora_alpha: int = 16
    roi_threshold: float = 0.5
    roi_pad: float = 0.06
    crop_margin: float = 0.1
    crop_min_frac: float = 0.12
    roi_min_pixels: int = 256 * 28 * 28
    roi_max_pixels: int = 2048 * 28 * 28
    # Access policy LoRA (trained in Stage II).
    policy_lora_layers: List[int] = field(default_factory=lambda: [24, 25, 26, 27])
    policy_lora_r: int = 16
    policy_lora_alpha: int = 32
    # Memory modules.
    global_rank: int = 64
    local_context_rank: int = 64
    selector_rank: int = 64
    conditioner_rank: int = 64
    set_context_rank: int = 64
    crop_scale: float = 1.0
    reader_hidden_scale: float = 1.0
    reader_memory_scale: float = 1.0
    relation: Dict = field(default_factory=lambda: {"rank": 128})

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class MemoryTypeEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.emb.weight, std=0.02)

    def stamp_local(self, memory: torch.Tensor) -> torch.Tensor:
        types = torch.full(memory.shape[:2], LOCAL, dtype=torch.long, device=memory.device)
        return memory + self.emb(types)

    def stamp_global(self, memory: torch.Tensor, image_tokens: int) -> torch.Tensor:
        types = torch.full(memory.shape[:2], GLOBAL, dtype=torch.long, device=memory.device)
        if 0 < image_tokens < memory.shape[1]:
            types[:, image_tokens:] = RELATION
        return memory + self.emb(types)


def _attach_lora(module: nn.Module, *, r: int, alpha: int, targets, adapter: str):
    from peft import LoraConfig, inject_adapter_in_model
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=list(targets))
    inject_adapter_in_model(config, module, adapter_name=adapter)


def load_image(path) -> Image.Image:
    """RGB image whose shorter side is at least 28 px."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if min(width, height) >= 28:
        return image
    scale = 28 / float(min(width, height))
    return image.resize((max(28, round(width * scale)), max(28, round(height * scale))), Image.LANCZOS)


def split_image_states(image_embeds: torch.Tensor, grid_thw, merge: int = 2) -> List[torch.Tensor]:
    """[V, d] merged visual tokens -> one [1, L_i, d] tensor per image."""
    states = image_embeds.unsqueeze(0)
    groups, offset = [], 0
    for t, h, w in grid_thw.tolist():
        count = t * h * w // (merge * merge)
        groups.append(states[:, offset:offset + count])
        offset += count
    return groups


@dataclass
class PromptState:
    """Per-question caches built at prefill and reused by every memory request."""
    images: List[Image.Image]
    image_paths: List[str]
    input_ids: torch.Tensor
    embeds: torch.Tensor
    grid: torch.Tensor
    position_ids: torch.Tensor
    image_states: List[torch.Tensor]
    prompt_hidden: torch.Tensor                     # request-layer hidden states of the prompt
    question_length: int                            # history length ending at the last question token
    relation_tokens: Optional[torch.Tensor] = None
    global_clauses: Optional[torch.Tensor] = None   # N <= 3: fact-writer clause queries [4, d]
    global_block: Optional[torch.Tensor] = None     # N <= 3: bridged block, reused
    global_high_n_raw: Optional[torch.Tensor] = None  # N >= 4: bridged raw block, re-read per request


@dataclass
class GlobalInputs:
    """Frozen inputs of a Global block; :meth:`ReMAP.weave_global` maps them to the block."""
    image_count: int
    visual: torch.Tensor                            # [V, d] visual tokens of all images
    image_ids: torch.Tensor                         # [V] image index of each visual token
    clauses: Optional[torch.Tensor] = None          # N <= 3: fact-writer clauses [4, d]
    summaries: Optional[torch.Tensor] = None        # N >= 4: per-image mean features [N, d]
    question: Optional[torch.Tensor] = None         # N >= 4: mean request-layer state of the prompt text [d]
    request_state: Optional[torch.Tensor] = None    # N >= 4: request-layer state before the request [1, d]


@dataclass
class LocalInputs:
    """Frozen inputs of a Local block (selected images and their re-encoded regions)."""
    positions: Tuple[int, ...]                      # selected images, strongest first
    regions: List[torch.Tensor]                     # per image: [1, R + 1, d] region tokens + mean image token
    bboxes: List[torch.Tensor]                      # per image: [1, 4] normalised box
    query_state: torch.Tensor                       # request-layer state before the request [1, d]
    boxes: List[tuple] = field(default_factory=list)  # pixel boxes (for inspection)
    query_length: int = 0                           # Twig query: history length up to the last content token


def image_inputs(image_states: Sequence[torch.Tensor]):
    """Concatenated visual tokens [V, d] (bf16) and their image indices [V]."""
    features = [s[0] for s in image_states]
    visual = torch.cat(features).to(dtype=torch.bfloat16)
    image_ids = torch.cat([torch.full((f.shape[0],), i, dtype=torch.long, device=f.device)
                           for i, f in enumerate(features)])
    return visual, image_ids


class ReMAP(nn.Module):
    def __init__(self, base, processor, config: ReMAPConfig, token_ids: MemoryTokenIds):
        super().__init__()
        self.config = config
        self.base = base
        self.processor = processor
        self.token_ids = token_ids
        d = config.hidden_size
        self.twig = TwigLocator(base.config, num_layers=config.twig_layers, branch_layer=config.twig_branch_layer)
        self.memory_rows = MemoryRows(token_ids, d)
        self.type_embedding = MemoryTypeEmbedding(d)
        self.global_memory = GlobalMemory(d, rank=config.global_rank)
        self.relation = RelationProducer(RelationConfig(hidden_size=d, **config.relation))
        self.local_memory = LocalMemory(d, selector_rank=config.selector_rank, context_rank=config.local_context_rank,
                                        conditioner_rank=config.conditioner_rank,
                                        set_context_rank=config.set_context_rank)
        _attach_lora(self.twig, r=config.twig_lora_r, alpha=config.twig_lora_alpha, targets=LORA_TARGETS,
                     adapter="locator")
        policy_targets = [f"layers.{i}.{'self_attn' if p in LORA_TARGETS[:4] else 'mlp'}.{p}"
                          for i in config.policy_lora_layers for p in LORA_TARGETS]
        _attach_lora(self.base.model, r=config.policy_lora_r, alpha=config.policy_lora_alpha,
                     targets=policy_targets, adapter="policy")
        self._forbidden = forbidden_response_ids(processor.tokenizer, token_ids)
        self.roi_processor = copy.copy(processor)
        roi_image = copy.deepcopy(processor.image_processor)
        roi_image.min_pixels, roi_image.max_pixels = config.roi_min_pixels, config.roi_max_pixels
        roi_image.size = {**getattr(roi_image, "size", {}), "shortest_edge": roi_image.min_pixels,
                          "longest_edge": roi_image.max_pixels}
        self.roi_processor.image_processor = roi_image

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, checkpoint_dir, *, base_model: Optional[str] = None, device="cuda",
                        dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        """Load the base model, register memory tokens, and load ReMAP's memory modules.

        ``checkpoint_dir`` holds ``remap_config.json``, ``memory.safetensors`` and the
        processor files.  Parameters keep the dtypes they were saved with.
        """
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        checkpoint_dir = Path(checkpoint_dir)
        config = ReMAPConfig.load(checkpoint_dir / "remap_config.json")
        base_path = base_model or config.base_model
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=dtype, attn_implementation=attn_implementation, device_map=device)
        processor = load_processor(base_path, checkpoint_dir)
        token_ids = register_memory_tokens(processor.tokenizer, base)
        model = cls(base, processor, config, token_ids)
        state = load_file(str(checkpoint_dir / "memory.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        missing = [k for k in missing if not k.startswith("base.") or ".policy." in k]
        if missing or unexpected:
            raise ValueError(f"memory checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        model.apply_scales()
        model.to(device)
        model.eval().requires_grad_(False)
        return model

    def apply_scales(self):
        weaver = self.local_memory.weaver
        weaver.crop_scale = self.config.crop_scale
        weaver.reader_hidden_scale = self.config.reader_hidden_scale
        weaver.reader_memory_scale = self.config.reader_memory_scale

    def memory_state_dict(self) -> Dict[str, torch.Tensor]:
        """Everything that is not a frozen base-model weight (policy LoRA included)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base.") or ".policy." in k}

    @property
    def device(self):
        return self.base.device

    # ------------------------------------------------------------------ token-level helpers
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.memory_rows.overlay_embeddings(self.base.model.embed_tokens(input_ids), input_ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.memory_rows.overlay_logits(self.base.lm_head(hidden), hidden).clone()
        logits[..., self._forbidden] = -float("inf")
        return logits

    def read_answer_hidden(self, hidden: torch.Tensor, reader_memory: Optional[torch.Tensor]) -> torch.Tensor:
        if reader_memory is None:
            return hidden
        return self.local_memory.weaver.read_decoder_hidden(hidden, reader_memory)

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill(self, image_paths: Sequence[str], content: list, fact_writer=None, question: str = ""):
        """Encode the prompt; build the per-question memory caches.

        Returns ``(state, output)`` where ``output`` is the decoder output with a KV cache.
        """
        images = [load_image(path) for path in image_paths]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True)
        encoded = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.device)
        ids = encoded["input_ids"]
        embeds = self.base.model.embed_tokens(ids)
        image_embeds = self.base.visual(encoded["pixel_values"].type(self.base.visual.dtype),
                                        grid_thw=encoded["image_grid_thw"]).to(embeds.device, embeds.dtype)
        mask = (ids == self.base.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(mask, image_embeds)
        grid = encoded["image_grid_thw"]
        position_ids, _ = self.base.get_rope_index(ids, grid, None, None, encoded["attention_mask"])
        output = self.base.model(inputs_embeds=embeds, position_ids=position_ids, use_cache=True,
                                 past_key_values=None, output_hidden_states=True)
        state = PromptState(
            images=images, image_paths=list(image_paths), input_ids=ids, embeds=embeds, grid=grid,
            position_ids=position_ids, image_states=split_image_states(image_embeds, grid),
            prompt_hidden=output.hidden_states[self.config.request_layer].detach(),
            question_length=last_question_index(ids[0].tolist(), self.processor.tokenizer,
                                                self.base.config.image_token_id) + 1)
        state.relation_tokens = self.relation_memory(state)
        if len(images) <= 3 and fact_writer is not None:
            _fact, clauses = fact_writer(image_paths, question)
            state.global_clauses = clauses.to(self.device)
        return state, output

    def relation_memory(self, state: PromptState) -> torch.Tensor:
        ids = state.input_ids[0]
        image_positions = torch.where(ids == self.base.config.image_token_id)[0]
        text = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        text[image_positions] = False
        image_ids, rows, cols = grid_ids(state.grid, ids.device)
        n = image_positions.numel()
        return self.relation(state.embeds[0, image_positions].unsqueeze(0), image_ids=image_ids[:n].unsqueeze(0),
                             row_ids=rows[:n].unsqueeze(0), col_ids=cols[:n].unsqueeze(0),
                             question_tokens=state.embeds[0, text].unsqueeze(0),
                             num_images=state.grid.shape[0]).to(dtype=state.embeds.dtype)

    # ------------------------------------------------------------------ memory blocks
    # ``global_inputs`` / ``locate`` gather the frozen inputs of a block; ``weave_*`` turn
    # them into the inserted block and are differentiable in the memory modules (Stage I).
    def global_inputs(self, state: PromptState, request_state: torch.Tensor) -> GlobalInputs:
        count = len(state.image_states)
        visual, image_ids = image_inputs(state.image_states)
        if count <= 3:
            if state.global_clauses is None:
                raise ValueError("Global memory for 1-3 images needs fact-writer clauses at prefill")
            return GlobalInputs(count, visual, image_ids, clauses=state.global_clauses)
        summaries = torch.stack([s[0].float().mean(dim=0) for s in state.image_states]).to(dtype=torch.bfloat16)
        text = state.input_ids != self.base.config.image_token_id
        question = state.prompt_hidden[text].float().mean(dim=0)
        return GlobalInputs(count, visual, image_ids, summaries=summaries, question=question,
                            request_state=request_state.detach())

    def _global_raw(self, inputs: GlobalInputs) -> torch.Tensor:
        """Type-stamped, bridged block before the request-time readout ([1, W, d])."""
        memory, count = self.global_memory, inputs.image_count
        if count <= 3:
            raw = memory.raw_low_n(inputs.visual, inputs.image_ids, inputs.clauses, count)
        else:
            raw = memory.raw_high_n(inputs.visual, inputs.image_ids, inputs.summaries, inputs.question, count)
        return memory.bridge(raw, count, self.type_embedding)

    def weave_global(self, inputs: GlobalInputs) -> torch.Tensor:
        block = self._global_raw(inputs)
        if inputs.image_count <= 3:
            return block
        return self.global_memory.readout(block, inputs.request_state)

    @torch.no_grad()
    def global_block(self, state: PromptState, request_state: torch.Tensor) -> torch.Tensor:
        """Inference: the N <= 3 block and the N >= 4 raw block are computed once per question."""
        if len(state.image_states) <= 3:
            if state.global_block is None:
                state.global_block = self._global_raw(self.global_inputs(state, request_state)).detach()
            return state.global_block
        if state.global_high_n_raw is None:
            state.global_high_n_raw = self._global_raw(self.global_inputs(state, request_state)).detach()
        return self.global_memory.readout(state.global_high_n_raw, request_state.detach()).detach()

    @torch.no_grad()
    def locate(self, state: PromptState, history, query_state: torch.Tensor, query_length: int) -> LocalInputs:
        """Select images, locate a region in each with Twig, crop and re-encode it."""
        local = self.local_memory
        embeds, positions, attention = history.prefix(query_length)
        twig_out = self.base.model(inputs_embeds=embeds, position_ids=positions, attention_mask=attention,
                                   use_cache=False, output_hidden_states=True)
        hidden = twig_out.hidden_states[self.config.twig_branch_layer]
        visual = (state.input_ids[0] == self.base.config.image_token_id).nonzero(as_tuple=True)[0]
        spans, offset = [], 0
        for t, h, w in state.grid.tolist():
            count = t * h * w // 4
            spans.append((offset, count, h // 2, w // 2))
            offset += count
        selection = local.selector(query_state, state.image_states)
        legacy_mean = torch.cat(state.image_states, dim=1).mean(1, keepdim=True)
        regions, bboxes, boxes = [], [], []
        for index in selection.positions:
            start, count, rows, cols = spans[index]
            heat = self.twig.heat(self.base.model, hidden, positions, [query_length - 1], visual[start:start + count],
                                  (rows, cols), attention_mask=attention)[0]
            image = state.images[index]
            box = relative_peak_bbox(heat, image.size[0], image.size[1], threshold=self.config.roi_threshold,
                                     pad=self.config.roi_pad)
            region, bbox = self._region_tokens(image, box, legacy_mean)
            boxes.append(box)
            regions.append(region)
            bboxes.append(bbox)
        return LocalInputs(selection.positions, regions, bboxes, query_state.detach(), boxes, query_length)

    def _region_tokens(self, image, box, legacy_mean):
        width, height = image.size
        x1, y1, x2, y2 = box
        bbox = torch.tensor([[x1 / width, y1 / height, x2 / width, y2 / height]], dtype=torch.float32,
                            device=self.device).clamp(0, 1)
        crop = crop_region(image, box, margin=self.config.crop_margin, min_frac=self.config.crop_min_frac)
        roi = self.roi_processor.image_processor(images=[crop], return_tensors="pt")
        roi_features = self.base.visual(roi["pixel_values"].to(self.device).type(self.base.visual.dtype),
                                        grid_thw=roi["image_grid_thw"].to(self.device))
        roi_features = roi_features.to(self.device, self.base.visual.dtype).unsqueeze(0)
        return torch.cat([roi_features, legacy_mean.to(roi_features.dtype)], dim=1), bbox

    def weave_local(self, inputs: LocalInputs, image_states: Sequence[torch.Tensor],
                    relation_tokens: torch.Tensor):
        """Returns ``(block [1, 32, d], reader_memory [1, 32, d])``."""
        local = self.local_memory
        weaver = local.weaver
        conditioned = local.conditioner.condition(inputs.query_state, relation_tokens.to(inputs.query_state.dtype))
        dtype = weaver.query_cond_proj.weight.dtype
        blocks, readers = [], []
        for index, region, bbox in zip(inputs.positions, inputs.regions, inputs.bboxes):
            context = weaver.image_context(image_states, anchor_index=index, roi_image_index=index)
            memory, crop_residual = weaver(region.to(dtype), query_state=conditioned.to(self.device, dtype),
                                           bbox=bbox, image_context=context.to(region.dtype).to(self.device, dtype))
            blocks.append(self.type_embedding.stamp_local(memory))
            readers.append(crop_residual)
        block, reader = pack_k32(blocks), pack_k32(readers)
        if len(blocks) > 1:
            selected = [image_states[i] for i in inputs.positions]
            block = local.conditioner.mix(blocks, selected, conditioned, block)
            reader = local.conditioner.mix(readers, selected, conditioned, reader)
        return local.set_context(block, image_states), reader

    @torch.no_grad()
    def local_block(self, state: PromptState, history, query_state: torch.Tensor, query_length: int):
        """Returns ``(block, reader_memory, inputs)``."""
        inputs = self.locate(state, history, query_state, query_length)
        block, reader = self.weave_local(inputs, state.image_states, state.relation_tokens)
        return block, reader, inputs


def last_question_index(ids: List[int], tokenizer, image_token_id: int) -> int:
    """Index of the last non-punctuation token of the user turn (before its ``<|im_end|>``)."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    user_end = [i for i, t in enumerate(ids) if t == im_end][-1]
    special = {int(v) for v in tokenizer.all_special_ids} | {int(image_token_id)}
    for index in range(user_end - 1, -1, -1):
        if ids[index] not in special and is_content_token(tokenizer, ids[index]):
            return index
    raise ValueError("user turn has no content token")


_PUNCTUATION = frozenset(".!?。！？,，;；:：'\"()[]{}-—…、·")


def is_content_token(tokenizer, token_id: int, excluded: Sequence[int] = ()) -> bool:
    if int(token_id) in {int(v) for v in excluded}:
        return False
    text = tokenizer.decode([int(token_id)]).strip()
    return bool(text) and not all(c in _PUNCTUATION for c in text)


def load_processor(base_path, checkpoint_dir=None):
    """Qwen2.5-VL processor with the base model's native image resolution and ReMAP's chat template."""
    from transformers import AutoImageProcessor, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_path, use_fast=True)
    processor.image_processor = AutoImageProcessor.from_pretrained(base_path, use_fast=True)
    template = Path(checkpoint_dir or "") / "chat_template.json"
    if checkpoint_dir is not None and template.exists():
        processor.chat_template = json.loads(template.read_text())["chat_template"]
    return processor
