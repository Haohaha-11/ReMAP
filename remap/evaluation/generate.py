"""Generate answers for benchmark questions (one process per GPU).

    torchrun --nproc_per_node 8 -m remap.evaluation.generate \
        --checkpoint ckpt/ --base-model Qwen/Qwen2.5-VL-7B-Instruct --data benchmarks/vstar.jsonl --output preds/vstar_on

Input rows (JSON lines): ``{"id", "benchmark", "images": [...], "question", "answer",
"options"?, ...}`` with the benchmark's original prompt.  The prompt is rewritten to ask
for reasoning followed by a final ``Answer:`` line (:func:`project_question`); images
are placed at the prompt's ``<image>`` placeholders.  Decoding is greedy with at most
``--max-calls`` memory reads; ``--no-memory`` disables reads (the OFF setting).
Each process writes ``predictions.rank{r}.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ..inference.session import DecodeConfig, generate, prompt_content
from ..modeling.fact_writer import FactWriter
from ..modeling.model import ReMAP
from ..scoring.benchmarks import generation_budget, project_question
from ..training.common import init_distributed, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--answer-grace", type=int, default=64)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    rank, world = init_distributed()
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    path = output / f"predictions.rank{rank:03d}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    model = ReMAP.from_pretrained(args.checkpoint, base_model=args.base_model)
    writer = None if args.no_memory else FactWriter.from_pretrained(args.checkpoint, base_model=args.base_model)
    with path.open("a") as stream:
        for row in rows[rank::world]:
            if row["id"] in done:
                continue
            question = project_question(row["question"])
            config = DecodeConfig(max_new_tokens=generation_budget(row["benchmark"]), max_calls=args.max_calls,
                                  answer_grace=args.answer_grace, allow_memory=not args.no_memory)
            started = time.time()
            try:
                result = generate(model, row["images"], prompt_content(question, len(row["images"])),
                                  question=question, fact_writer=writer, config=config)
                record = dict(id=row["id"], benchmark=row["benchmark"], status="success", prediction=result.text,
                              token_ids=result.token_ids, stop_reason=result.stop_reason,
                              calls=[dict(kind=e.kind, step=e.step, images=list(e.images or ()),
                                          boxes=[list(map(float, b)) for b in (e.boxes or [])]) for e in result.events])
            except Exception:
                record = dict(id=row["id"], benchmark=row["benchmark"], status="error", prediction="",
                              error=traceback.format_exc()[-2000:])
            record["seconds"] = time.time() - started
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
