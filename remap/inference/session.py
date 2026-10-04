"""Decoding sessions with memory requests (shared by evaluation and RL rollouts).

A :class:`PromptCache` holds the prefill of one question (KV cache, history, memory
caches).  :meth:`Session.fork` starts a fresh trajectory from it by cropping the KV
cache back to the prompt.  A session can teacher-force a prefix (re-inserting the
memory of earlier requests), and :meth:`Session.decode` continues it, either greedily
or by sampling, optionally recording routing probabilities for branch selection.

Decoding rules:

* END tokens are never produced; every other special token except EOS is banned.
* Request tokens are banned once the call budget is used or the answer has begun.
* After ``Answer:`` appears, at most ``answer_grace`` further tokens are generated.
* After a Local request, answer-time hidden states are adapted by the reader memory
  of the latest Local request.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from ..modeling.memory_tokens import logit_mask
from ..modeling.model import PromptState, ReMAP, is_content_token

_ANSWER = re.compile(r"[Aa]nswer\s*:")
HARD_CALL_LIMIT = 5


def answer_committed(tokenizer, ids: Sequence[int]) -> bool:
    return bool(_ANSWER.search(tokenizer.decode(list(ids)[-8:], skip_special_tokens=True)))


@dataclass
class DecodeConfig:
    max_new_tokens: int = 2048
    max_calls: int = 4
    answer_grace: int = 64
    temperature: float = 0.0
    allow_memory: bool = True
    record_routes: bool = False        # store p(<mem_g>), p(<mem_l>) at every decision (branch selection)
    record_logprobs: bool = False      # store the sampled token log-probabilities


@dataclass
class RouteState:
    step: int
    p_global: float
    p_local: float
    action: str                        # defer / global / local


@dataclass
class MemoryEvent:
    kind: str
    step: int
    block: torch.Tensor
    reader: Optional[torch.Tensor] = None
    boxes: Optional[list] = None
    images: Optional[tuple] = None
    inputs: Optional[object] = None    # GlobalInputs / LocalInputs, kept when the session captures them


@dataclass
class Trajectory:
    token_ids: List[int]
    events: List[MemoryEvent]
    stop_reason: str
    answer_commit_step: Optional[int] = None
    route_states: List[RouteState] = field(default_factory=list)
    logprobs: List[torch.Tensor] = field(default_factory=list)
    root: int = -1
    variant: int = 0
    branch_step: Optional[int] = None
    reward: Optional[float] = None
    text: str = ""

    @property
    def blocks(self):
        return [e.block for e in self.events]

    @property
    def kinds(self):
        return [e.kind for e in self.events]


class History:
    """Every embedding fed to the decoder (prompt, tokens, latent blocks) with its positions."""

    def __init__(self, embeds, position_ids, attention_mask):
        self.embeds, self.position_ids, self.attention_mask = embeds.detach(), position_ids.detach(), attention_mask.detach()

    def copy(self) -> "History":
        return History(self.embeds, self.position_ids, self.attention_mask)

    @property
    def length(self) -> int:
        return int(self.embeds.shape[1])

    def append(self, embeds, position_ids):
        self.embeds = torch.cat([self.embeds, embeds.detach()], dim=1)
        self.position_ids = torch.cat([self.position_ids, position_ids.detach()], dim=2)
        ones = torch.ones((self.attention_mask.shape[0], embeds.shape[1]), device=self.attention_mask.device,
                          dtype=self.attention_mask.dtype)
        self.attention_mask = torch.cat([self.attention_mask, ones], dim=1)

    def prefix(self, length: int):
        return self.embeds[:, :length], self.position_ids[:, :, :length], self.attention_mask[:, :length]


@dataclass
class SegmentCursor:
    """History length at the last content (non-punctuation) token of the current segment."""
    history_length: int
    fallback_length: int = 0
    has_content: bool = True

    def __post_init__(self):
        self.fallback_length = self.fallback_length or self.history_length

    def copy(self) -> "SegmentCursor":
        return SegmentCursor(self.history_length, self.fallback_length, self.has_content)

    def observe(self, tokenizer, token_id: int, history_length: int, excluded: Sequence[int]):
        self.fallback_length = history_length
        if is_content_token(tokenizer, token_id, excluded):
            self.history_length = history_length
            self.has_content = True

    def query_length(self) -> int:
        return self.history_length if self.has_content else self.fallback_length


@dataclass
class PromptCache:
    state: PromptState
    output: object
    cache_length: int
    history: History


def _position(next_pos: int, width: int, device) -> torch.Tensor:
    return torch.arange(next_pos, next_pos + width, device=device).view(1, 1, -1).expand(3, 1, -1)


class Session:
    def __init__(self, model: ReMAP, prompt: PromptCache, *, capture_inputs: bool = False):
        self.model = model
        self.capture_inputs = capture_inputs
        self.on_request = None             # callback(session, token) before a request is served
        self.prompt = prompt
        self.state = prompt.state
        tokenizer = model.processor.tokenizer
        self.tokenizer = tokenizer
        ids = model.token_ids
        self.ids = ids
        prompt.output.past_key_values.crop(prompt.cache_length)
        self.output = prompt.output
        self.past = prompt.output.past_key_values
        self.history = prompt.history.copy()
        self.next_pos = int(self.state.position_ids.max()) + 1
        self.request_state = prompt.output.hidden_states[model.config.request_layer][:, -1].detach()
        self.cursor = SegmentCursor(self.state.question_length)
        self.reader_memory: Optional[torch.Tensor] = None
        self.local_calls = 0
        self.events: List[MemoryEvent] = []
        self.tokens: List[int] = []
        self.answer_seen = False
        vocab = model.base.lm_head.weight.shape[0]
        dtype = prompt.output.last_hidden_state.dtype
        self.ban_end = logit_mask(vocab, ids.end_ids, device=model.device, dtype=dtype)
        self.ban_request = logit_mask(vocab, ids.request_ids, device=model.device, dtype=dtype)
        self.eos = int(tokenizer.eos_token_id)

    # -------------------------------------------------------------- construction
    @classmethod
    @torch.no_grad()
    def prefill(cls, model: ReMAP, image_paths, content, *, question: str, fact_writer=None) -> PromptCache:
        state, output = model.prefill(image_paths, content, fact_writer=fact_writer, question=question)
        history = History(state.embeds, state.position_ids, torch.ones_like(state.input_ids))
        return PromptCache(state, output, int(output.past_key_values.get_seq_length()), history)

    # -------------------------------------------------------------- stepping
    @torch.no_grad()
    def advance(self, token: int):
        """Feed one generated token; serve a memory request if it is one."""
        model, ids = self.model, self.ids
        kind = ids.kind(token)
        if kind is not None and self.on_request is not None:
            self.on_request(self, token)
        query_length = self.cursor.query_length() if kind == "local" else None
        pre_request = self.request_state
        embedding = model.embed(torch.tensor([[token]], device=model.device))
        position = _position(self.next_pos, 1, model.device)
        self.output = model.base.model(inputs_embeds=embedding, position_ids=position, past_key_values=self.past,
                                       use_cache=True, output_hidden_states=True)
        self.history.append(embedding, position)
        self.past, self.next_pos = self.output.past_key_values, self.next_pos + 1
        self.tokens.append(int(token))
        if kind is None:
            self.cursor.observe(self.tokenizer, token, self.history.length, ids.all_ids)
        else:
            step = len(self.tokens) - 1
            if kind == "global":
                block = model.global_block(self.state, pre_request)
                inputs = model.global_inputs(self.state, pre_request) if self.capture_inputs else None
                event = MemoryEvent("global", step, block, inputs=inputs)
            else:
                block, reader, inputs = model.local_block(self.state, self.history, pre_request, query_length)
                self.reader_memory = reader.detach()
                self.local_calls += 1
                event = MemoryEvent("local", step, block.detach(), reader=self.reader_memory, boxes=inputs.boxes,
                                    images=inputs.positions, inputs=inputs if self.capture_inputs else None)
            end = model.embed(torch.tensor([[ids.end_for(token)]], device=model.device))
            sequence = torch.cat([block.to(end.dtype), end], dim=1)
            position = _position(self.next_pos, sequence.shape[1], model.device)
            self.output = model.base.model(inputs_embeds=sequence, position_ids=position, past_key_values=self.past,
                                           use_cache=True, output_hidden_states=True)
            self.history.append(sequence, position)
            self.past, self.next_pos = self.output.past_key_values, self.next_pos + sequence.shape[1]
            self.events.append(event)
            self.cursor.has_content = False
        self.request_state = self.output.hidden_states[model.config.request_layer][:, -1].detach()

    @torch.no_grad()
    def replay(self, prefix: Sequence[int]):
        for token in prefix:
            self.advance(int(token))

    def checkpoint(self):
        """Snapshot after a replayed prefix so several continuations can be sampled from it."""
        return dict(cache_length=int(self.past.get_seq_length()), output=self.output, next_pos=self.next_pos,
                    history=self.history.copy(), request_state=self.request_state, cursor=self.cursor.copy(),
                    reader=self.reader_memory, local_calls=self.local_calls, events=list(self.events),
                    tokens=list(self.tokens))

    def restore(self, snapshot):
        self.past.crop(snapshot["cache_length"])
        self.output, self.next_pos = snapshot["output"], snapshot["next_pos"]
        self.history = snapshot["history"].copy()
        self.request_state, self.cursor = snapshot["request_state"], snapshot["cursor"].copy()
        self.reader_memory, self.local_calls = snapshot["reader"], snapshot["local_calls"]
        self.events, self.tokens = list(snapshot["events"]), list(snapshot["tokens"])
        self.answer_seen = False

    # -------------------------------------------------------------- decoding
    def _logits(self, cfg: DecodeConfig, answer_active: bool):
        model = self.model
        hidden = self.output.last_hidden_state
        self.answer_seen = self.answer_seen or answer_committed(self.tokenizer, self.tokens)
        if self.local_calls > 0 and self.answer_seen:
            hidden = model.read_answer_hidden(hidden, self.reader_memory)
        raw = model.logits(hidden[:, -1:]) + self.ban_end
        if len(self.events) >= HARD_CALL_LIMIT:
            raw = raw.clone()
            raw[..., self.ids.request_ids] = -torch.inf
        banned = (not cfg.allow_memory) or answer_active or (cfg.max_calls > 0 and len(self.events) >= cfg.max_calls)
        policy = raw + self.ban_request if banned else raw
        return policy[0, -1], raw[0, -1]

    def _route(self, raw_row) -> tuple:
        probs = torch.softmax(raw_row.float(), dim=-1)
        return float(probs[self.ids.mem_g]), float(probs[self.ids.mem_l])

    @staticmethod
    def _pick(row, temperature: float):
        if temperature and temperature > 0:
            probs = torch.softmax(row.float() / float(temperature), dim=-1)
            token = int(torch.multinomial(probs, 1))
            return token, probs[token].log().detach()
        token = int(row.argmax(-1))
        return token, torch.log_softmax(row.float(), dim=-1)[token].detach()

    @torch.no_grad()
    def decode(self, cfg: DecodeConfig, max_steps: Optional[int] = None) -> Trajectory:
        """Continue from the current state; returns the newly generated part."""
        start_tokens = len(self.tokens)
        start_events = len(self.events)
        answer_active = answer_committed(self.tokenizer, self.tokens) if self.tokens else False
        grace = cfg.answer_grace if answer_active else -1
        commit_step = None
        routes: List[RouteState] = []
        logprobs: List[torch.Tensor] = []
        stop = "max_new_tokens"
        budget = cfg.max_new_tokens if max_steps is None else max_steps
        row, raw = self._logits(cfg, answer_active)
        pending = self._route(raw) if cfg.record_routes else None
        token, logp = self._pick(row, cfg.temperature)
        for step in range(budget):
            if token == self.eos:
                if cfg.record_routes and not answer_active:
                    routes.append(RouteState(len(self.tokens) - start_tokens, *pending, "defer"))
                stop = "eos"
                break
            generated = self.tokens[start_tokens:] + [token]
            committed = not answer_active and answer_committed(self.tokenizer, self.tokens + [token])
            if cfg.record_logprobs:
                logprobs.append(logp)
            kind = self.ids.kind(token)
            if kind is not None and answer_active:
                raise RuntimeError("memory request after the answer was committed")
            if cfg.record_routes and (kind is not None or (not answer_active and not committed)):
                routes.append(RouteState(len(generated) - 1, *pending, kind or "defer"))
            self.advance(token)
            if committed:
                answer_active, grace, commit_step = True, cfg.answer_grace, len(generated) - 1
            elif answer_active:
                grace -= 1
                if grace <= 0:
                    stop = "answer"
                    break
            row, raw = self._logits(cfg, answer_active)
            pending = self._route(raw) if cfg.record_routes else None
            token, logp = self._pick(row, cfg.temperature)
        return Trajectory(self.tokens[start_tokens:], self.events[start_events:], stop, commit_step,
                          routes, logprobs)


def prompt_content(question: str, image_count: int) -> list:
    """Chat content: images at the question's ``<image>`` placeholders, else all images first."""
    parts = question.split("<image>")
    if len(parts) - 1 != image_count:
        return [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": question}]
    content = []
    for index, text in enumerate(parts):
        if text:
            content.append({"type": "text", "text": text})
        if index < image_count:
            content.append({"type": "image"})
    return content


@torch.no_grad()
def generate(model: ReMAP, image_paths, content, *, question: str, fact_writer=None,
             config: DecodeConfig = DecodeConfig()) -> Trajectory:
    """Greedy (or sampled) decoding of one question with memory requests."""
    session = Session(model, Session.prefill(model, image_paths, content, question=question, fact_writer=fact_writer))
    trajectory = session.decode(config)
    trajectory.text = model.processor.tokenizer.decode(trajectory.token_ids, skip_special_tokens=True)
    return trajectory
