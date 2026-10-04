"""Collect Stage I states: decode training questions with the current model and keep the
student's state at every memory request.

    torchrun --nproc_per_node 8 -m remap.training.stage1.collect \
        --checkpoint ckpt/ --base-model Q-Zoom --questions train.jsonl --output states/ [--mode natural]

Modes:

* ``natural``: the model decides when and what to read (greedy, at most ``--max-calls``).
* ``global_after``: questions with four or more images only; requests are disabled and one
  Global request is forced after ``--global-after`` generated tokens, unless the answer
  has already started.  This adds Global states for the cases the student rarely reads.

For the teacher, every state also records pixel proxies of its reads: the exact crops
of a Local read and the full images for a Global read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ...inference.session import DecodeConfig, Session, answer_committed, prompt_content
from ...modeling.fact_writer import FactWriter
from ...modeling.local_memory import crop_region
from ...modeling.model import ReMAP
from ..common import init_distributed
from .states import MemoryState, pack_inputs, save_state


def _read_views(model, prompt, event, folder: Path, name: str):
    if event.kind == "global":
        return [dict(path=path, image_index=i) for i, path in enumerate(prompt.state.image_paths)]
    views = []
    for image_index, box in zip(event.inputs.positions, event.inputs.boxes):
        path = folder / f"{name}_crop{image_index}.png"
        crop_region(prompt.state.images[image_index], box, margin=model.config.crop_margin,
                    min_frac=model.config.crop_min_frac).save(path)
        views.append(dict(path=str(path), image_index=int(image_index)))
    return views


@torch.no_grad()
def collect_question(model, writer, row: dict, folder: Path, *, mode: str, max_new_tokens: int, max_calls: int,
                     global_after: int):
    images, question = row["images"], row["question"]
    if mode == "global_after" and len(images) < 4:
        return []
    tokenizer = model.processor.tokenizer
    prompt = Session.prefill(model, images, prompt_content(question, len(images)), question=question,
                             fact_writer=writer)
    session = Session(model, prompt, capture_inputs=True)
    snapshots = []

    def on_request(s, _token):
        snapshots.append(dict(prefix=list(s.tokens), embeds=s.history.embeds.cpu(), positions=s.history.position_ids.cpu(),
                              next_position=s.next_pos,
                              previous_reader=None if s.reader_memory is None else s.reader_memory.cpu()))

    session.on_request = on_request
    if mode == "natural":
        session.decode(DecodeConfig(max_new_tokens=max_new_tokens, max_calls=max_calls))
    else:
        part = session.decode(DecodeConfig(max_new_tokens=global_after, allow_memory=False))
        if part.stop_reason != "max_new_tokens" or answer_committed(tokenizer, session.tokens):
            return []
        session.advance(model.token_ids.mem_g)
    folder.mkdir(parents=True, exist_ok=True)
    records, previous_reads = [], []
    for ordinal, (snapshot, event) in enumerate(zip(snapshots, session.events)):
        state_id = f"{row['id'].replace(':', '_')}_c{ordinal}"
        state = MemoryState(kind=event.kind, question=question, images=list(images),
                            prompt_ids=prompt.state.input_ids[0].cpu(), grid=prompt.state.grid.cpu(),
                            prefix_ids=snapshot["prefix"], embeds=snapshot["embeds"], positions=snapshot["positions"],
                            next_position=snapshot["next_position"], relation_tokens=prompt.state.relation_tokens.cpu(),
                            inputs=pack_inputs(event.inputs), previous_reader=snapshot["previous_reader"],
                            state_id=state_id)
        save_state(folder / f"{state_id}.pt", state)
        current = dict(kind=event.kind, views=_read_views(model, prompt, event, folder, state_id))
        records.append(dict(state_id=state_id, row_id=row["id"], kind=event.kind, state=str(folder / f"{state_id}.pt"),
                            question=question, images=list(images),
                            prefix_text=tokenizer.decode(snapshot["prefix"], skip_special_tokens=True),
                            previous_reads=list(previous_reads), current_read=current))
        previous_reads.append(current)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("natural", "global_after"), default="natural")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--global-after", type=int, default=32)
    args = parser.parse_args()
    rank, world = init_distributed()
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()][rank::world]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = ReMAP.from_pretrained(args.checkpoint, base_model=args.base_model)
    writer = FactWriter.from_pretrained(args.checkpoint, base_model=args.base_model)
    with (output / f"states.rank{rank:03d}.jsonl").open("a") as stream:
        for row in rows:
            for record in collect_question(model, writer, row, output / row["id"].replace(":", "_"), mode=args.mode,
                                           max_new_tokens=args.max_new_tokens, max_calls=args.max_calls,
                                           global_after=args.global_after):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()


if __name__ == "__main__":
    main()
