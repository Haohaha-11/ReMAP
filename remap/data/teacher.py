"""Client for the teacher model (an OpenAI-compatible multimodal chat endpoint).

We used Qwen3.5-397B-A17B through DashScope's OpenAI-compatible API: temperature 0,
thinking disabled, at most 1400 output tokens, JSON-object responses.

Configuration (environment): ``TEACHER_API_URL`` (base URL, e.g.
``https://dashscope.aliyuncs.com/compatible-mode/v1``), ``TEACHER_API_KEY`` and
optionally ``TEACHER_MODEL``.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

from PIL import Image

DEFAULT_MODEL = "qwen3.5-397b-a17b"
# Student image budgets (pixels): overview images and Local crops are shown to the teacher
# at the resolution the student itself encodes them.
BUDGETS = {"main": (50176, 802816), "roi": (256 * 784, 2048 * 784)}


class RateLimiter:
    """At most ``rps`` requests per second and ``token_budget`` estimated tokens per 30 s."""

    def __init__(self, rps: float, token_budget: int = 450000):
        self.interval, self.token_budget = 1.0 / rps, token_budget
        self.lock, self.last, self.tokens = threading.Lock(), 0.0, deque()

    def wait(self, estimated_tokens: int = 0):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.tokens and now - self.tokens[0][0] > 30:
                    self.tokens.popleft()
                if (now - self.last >= self.interval
                        and sum(n for _, n in self.tokens) + estimated_tokens <= self.token_budget):
                    self.last = now
                    self.tokens.append((now, estimated_tokens))
                    return
            time.sleep(0.05)


def image_view(path, *, budget: str, box=None, save_to: Optional[Path] = None) -> dict:
    """A chat ``image_url`` part: the image (or a box of it) resized to the student budget, JPEG."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if box is not None:
        width, height = image.size
        image = image.crop((math.floor(box[0] * width), math.floor(box[1] * height),
                            math.ceil(box[2] * width), math.ceil(box[3] * height)))
    if min(image.size) < 28:
        scale = 28 / min(image.size)
        image = image.resize(tuple(max(28, round(side * scale)) for side in image.size), Image.Resampling.LANCZOS)
    minimum, maximum = BUDGETS[budget]
    height, width = smart_resize(image.height, image.width, factor=28, min_pixels=minimum, max_pixels=maximum)
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    data = buffer.getvalue()
    if save_to is not None:
        Path(save_to).write_bytes(data)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}


def parse_json(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean)
    value = json.loads(clean)
    if not isinstance(value, dict):
        raise ValueError("teacher must return a JSON object")
    return value


class TeacherClient:
    def __init__(self, *, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 rps: float = 9.5, token_budget: int = 450000, max_tokens: int = 1400, attempts: int = 3):
        self.model = model or os.environ.get("TEACHER_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ["TEACHER_API_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["TEACHER_API_KEY"]
        self.rate = RateLimiter(rps, token_budget)
        self.max_tokens, self.attempts = max_tokens, attempts
        self.usage, self.lock = Counter(), threading.Lock()

    def _estimate(self, messages) -> int:
        estimate = self.max_tokens
        for message in messages:
            parts = message["content"] if isinstance(message["content"], list) else [{"type": "text", "text": message["content"]}]
            for part in parts:
                if part["type"] == "text":
                    estimate += max(1, len(part["text"]) // 3)
                else:
                    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    with Image.open(io.BytesIO(raw)) as image:
                        estimate += math.ceil(image.width / 28) * math.ceil(image.height / 28) * 2
        return estimate

    def call(self, system: str, images: list, payload: dict, log_path: Optional[Path] = None, *,
             payload_first: bool = False) -> dict:
        """One JSON-object completion; the user turn is ``images`` and ``payload`` as JSON text."""
        import requests

        text = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": text + images if payload_first else images + text}]
        body = dict(model=self.model, messages=messages, temperature=0, enable_thinking=False,
                    max_tokens=self.max_tokens, response_format={"type": "json_object"})
        estimate = self._estimate(messages)
        last_error = ""
        for attempt in range(self.attempts):
            self.rate.wait(estimate)
            try:
                response = requests.post(self.base_url + "/chat/completions", json=body, timeout=300,
                                         headers={"Authorization": "Bearer " + self.api_key})
            except requests.RequestException as error:
                last_error = type(error).__name__
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                time.sleep(4 * 2 ** attempt)
                continue
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            if log_path is not None:
                Path(log_path).write_text(json.dumps(dict(payload=payload, response=result), ensure_ascii=False, indent=1))
            if choice.get("finish_reason") != "stop":
                last_error = "truncated response"
                continue
            try:
                value = parse_json(choice["message"]["content"])
            except (ValueError, TypeError):
                # Ask once more for the JSON object only; the task itself is unchanged.
                body["messages"] = messages + [{"role": "user", "content": "Return the requested JSON object only."}]
                last_error = "invalid JSON"
                continue
            with self.lock:
                self.usage.update({k: v for k, v in result.get("usage", {}).items() if isinstance(v, int)})
            return value
        raise RuntimeError(f"teacher request failed: {last_error}")
