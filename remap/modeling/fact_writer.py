"""Fact writer: clause queries for Global memory when there are 1 to 3 images.

The writer is a separate Qwen2.5-VL model with LoRA adapters on the decoder and on
the vision tower.  It never sees options or answers.  Given the images and the
question stem it writes one structured line, e.g.::

    I1{red mug on a desk}; I2{same mug, handle to the left}; REL{I2 is I1 rotated}.

The line is then teacher-forced through the writer once more, and the last-layer
hidden states of the tokens inside ``I1{..}``, ``I2{..}``, ``I3{..}`` and ``REL{..}``
are averaged into four clause vectors (missing slots stay zero).
"""
from __future__ import annotations

import itertools
import re
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

OPTION_BLOCK = re.compile(r"\n\s*(?:\([A-H]\)|[A-H][.)])\s+", re.IGNORECASE)
CLAUSE_SLOTS = ("I1", "I2", "I3", "REL")
MAX_NEW_TOKENS = 64


def question_stem(question: str) -> str:
    """Drop the option block so the writer cannot copy or rank options."""
    return re.sub(r"\s+", " ", OPTION_BLOCK.split(str(question), maxsplit=1)[0]).strip()


def schema(n_images: int) -> str:
    return "; ".join([*(f"I{i}{{...}}" for i in range(1, n_images + 1)), "REL{...}"]) + "."


def instruction(n_images: int, question: str) -> str:
    forbidden = f"I{n_images + 1}"
    return (
        f"There are exactly {n_images} ordered image(s). Image count is binding.\n"
        f"Output exactly this {n_images}-image schema and nothing else:\n{schema(n_images)}\n"
        f"Replace only the text inside braces. Emit I1 through I{n_images} exactly once, "
        f"in order, followed by REL exactly once. Never emit {forbidden} or any extra slot. "
        "Begin with I1{. After the final period, emit EOS immediately. Do not stop before "
        "all required image slots and REL are complete. Image indices are binding: never "
        "move evidence between images. Do not answer the question, choose an image, copy an "
        "option, or emit 'Observed:'. Use unknown inside a required clause when evidence is "
        "unavailable.\n\nDownstream question:\n"
        f"{question_stem(question)}"
    )


def min_side(image: Image.Image, minimum: int = 28) -> Image.Image:
    width, height = image.size
    if min(width, height) >= minimum:
        return image
    scale = minimum / min(width, height)
    return image.resize((max(minimum, round(width * scale)), max(minimum, round(height * scale))),
                        Image.Resampling.BICUBIC)


# ------------------------------------------------------------------ text post-processing
def canonicalize(fact: str, n_images: int) -> str:
    """Force exactly ``I1..IN`` then ``REL`` slots, in order, with sanitised payloads."""
    labels = [*(f"I{i}" for i in range(1, n_images + 1)), "REL"]

    def sanitize(value: str) -> str:
        value = re.sub(r"[{}\r\n]+", " ", str(value))
        value = re.sub(r"(?i)\bobserved\b", "evidence", value)
        value = re.sub(r"\s+", " ", value).strip(" ;,.")
        return value or "unknown"

    text = str(fact).strip()
    found = {}
    for index, label in enumerate(labels):
        match = re.search(rf"(?:^|;\s*){label}\{{", text)
        if not match:
            continue
        start = match.end()
        nexts = []
        for other in labels[index + 1:]:
            following = re.search(rf";\s*{other}\{{", text[start:])
            if following:
                nexts.append(start + following.start())
        chunk = text[start:min(nexts) if nexts else len(text)]
        chunk = chunk.rsplit("}", 1)[0] if "}" in chunk else chunk
        found[label] = sanitize(chunk)
    return "; ".join(f"{label}{{{found.get(label, 'unknown')}}}" for label in labels) + "."


_PARTIAL_STRIP = re.compile(r"I1\{strip\.rank=[123]\}.*I2\{strip\.rank=[123]\}.*I3\{strip\.rank=[123]\}", re.DOTALL)


def _border_cost(first: Image.Image, second: Image.Image, axis: str) -> float:
    a = np.asarray(first.convert("RGB"), dtype=np.float32)
    b = np.asarray(second.convert("RGB"), dtype=np.float32)
    edge_a, edge_b = (a[-1, :, :], b[0, :, :]) if axis == "top_to_bottom" else (a[:, -1, :], b[:, 0, :])

    def resample(edge, samples=128):
        if edge.shape[0] == samples:
            return edge
        source = np.linspace(0.0, 1.0, edge.shape[0], dtype=np.float32)
        target = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        return np.stack([np.interp(target, source, edge[:, c]) for c in range(3)], axis=1).astype(np.float32)

    return float(np.mean((resample(edge_a) - resample(edge_b)) ** 2) / (255.0 ** 2))


def strip_fact(images: Sequence[Image.Image]) -> str:
    """Order three strips of one picture by border continuity (pixels only)."""
    ratios = [float(im.width) / max(float(im.height), 1.0) for im in images]
    axis = "top_to_bottom" if sum(ratios) / len(ratios) >= 1.0 else "left_to_right"
    costs = {(i, j): _border_cost(images[i], images[j], axis)
             for i in range(3) for j in range(3) if i != j}
    ranked = sorted((sum(costs[(c[k], c[k + 1])] for k in range(2)), c) for c in itertools.permutations(range(3)))
    ranks = [0, 0, 0]
    for rank, index in enumerate(ranked[0][1], start=1):
        ranks[index] = rank
    return (f"I1{{strip.rank={ranks[0]}}}; I2{{strip.rank={ranks[1]}}}; "
            f"I3{{strip.rank={ranks[2]}}}; REL{{axis={axis}}}.")


def correct_strips(fact: str, images: Sequence[Image.Image]) -> str:
    """For three elongated images (or a strip-rank answer), recompute the strip order from pixels."""
    if len(images) != 3:
        return fact
    elongated = all(max(float(im.width), float(im.height)) / max(min(float(im.width), float(im.height)), 1.0) >= 1.505
                    for im in images)
    if _PARTIAL_STRIP.search(str(fact)) or elongated:
        return strip_fact(images)
    return fact


def _slot_ranges(fact: str, n_images: int) -> dict:
    labels = [*(f"I{i}" for i in range(1, n_images + 1)), "REL"]
    result, cursor = {}, 0
    for label in labels:
        match = re.search(rf"{label}\{{([^{{}}]+)\}}", fact[cursor:])
        if match is None:
            raise ValueError(f"fact is missing slot {label}: {fact!r}")
        result[label] = (cursor + match.start(1), cursor + match.end(1))
        cursor += match.end(0)
    return result


class FactWriter:
    """Wraps a Qwen2.5-VL model carrying the writer LoRA adapters."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, checkpoint_dir, *, base_model: str, device="cuda", dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2"):
        """``checkpoint_dir`` holds ``fact_writer.safetensors`` and ``fact_writer_config.json``."""
        import json
        from pathlib import Path

        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        checkpoint_dir = Path(checkpoint_dir)
        cfg = json.loads((checkpoint_dir / "fact_writer_config.json").read_text())
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype, attn_implementation=attn_implementation)
        inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0, bias="none",
                                           target_modules=cfg["decoder_targets"]), model.model, adapter_name="writer")
        inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0, bias="none",
                                           target_modules=cfg["vision_targets"]), model.visual,
                                adapter_name="writer_vision")
        result = model.load_state_dict(load_file(str(checkpoint_dir / "fact_writer.safetensors")), strict=False)
        missing = [k for k in result.missing_keys if ".writer" in k]
        if missing or result.unexpected_keys:
            raise ValueError(f"fact writer mismatch: missing={missing[:4]} unexpected={result.unexpected_keys[:4]}")
        processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
        template = checkpoint_dir / "chat_template.json"
        if template.exists():
            processor.chat_template = json.loads(template.read_text())["chat_template"]
        return cls(model.eval().to(device), processor)

    def _encode(self, images: List[Image.Image], question: str, fact: str | None = None):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction(len(images), question)}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True)
        text = prompt if fact is None else prompt + fact.strip() + self.processor.tokenizer.eos_token
        encoded = self.processor(text=[text], images=images, padding=fact is None, return_tensors="pt")
        prompt_length = None
        if fact is not None:
            prompt_length = int(self.processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1])
        return encoded, prompt_length

    @torch.inference_mode()
    def write(self, image_paths: Sequence[str], question: str) -> str:
        """Greedy structured fact (at most 64 tokens), canonicalised to the exact schema."""
        images = [min_side(Image.open(path).convert("RGB")) for path in image_paths]
        encoded, _ = self._encode(images, question)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        eos = int(self.processor.tokenizer.eos_token_id)
        generated = self.model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        use_cache=True, eos_token_id=eos, pad_token_id=eos)
        ids = generated[0, encoded["input_ids"].shape[1]:].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        raw_images = [Image.open(path).convert("RGB") for path in image_paths]
        text = correct_strips(text, raw_images)
        return canonicalize(text, len(image_paths))

    @torch.inference_mode()
    def clause_queries(self, image_paths: Sequence[str], question: str, fact: str) -> torch.Tensor:
        """Average last-layer hidden states over each slot's tokens -> [4, d] (bfloat16, CPU)."""
        images = [min_side(Image.open(path).convert("RGB")) for path in image_paths]
        encoded, prompt_length = self._encode(images, question, fact)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        hidden = self.model(**encoded, use_cache=False, output_hidden_states=True).hidden_states[-1]
        offsets = self.processor.tokenizer(fact, add_special_tokens=False,
                                           return_offsets_mapping=True)["offset_mapping"]
        positions = torch.arange(prompt_length, encoded["input_ids"].shape[1], device=hidden.device)
        ranges = _slot_ranges(fact, len(image_paths))
        queries = torch.zeros(len(CLAUSE_SLOTS), hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
        for slot_index, slot in enumerate(CLAUSE_SLOTS):
            if slot not in ranges:
                continue
            start, end = ranges[slot]
            tokens = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
            queries[slot_index] = hidden[0, positions[torch.tensor(tokens, device=hidden.device)]].mean(dim=0)
        return queries.to(dtype=torch.bfloat16, device="cpu")

    def __call__(self, image_paths: Sequence[str], question: str) -> Tuple[str, torch.Tensor]:
        fact = self.write(image_paths, question)
        return fact, self.clause_queries(image_paths, question, fact)
