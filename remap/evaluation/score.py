"""Score generated answers.

    python -m remap.evaluation.score --data benchmarks/vstar.jsonl --predictions preds/vstar_on [--judge]

Every benchmark question must have exactly one prediction; failed generations score 0.
Multiple-choice benchmarks use their official answer parsers, MIMIC tasks the released
MIMIC metrics on the raw output.  MM-Vet and MathVista free-form answers need an
LLM judge (``--judge``; see :mod:`remap.scoring.judge`), otherwise they are reported as
pending.  Grouped benchmarks need their group field in the data rows: ``source``
(CV-Bench 2D), ``sub_task`` (BLINK) or ``cycle_category`` (HR-Bench).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..scoring.benchmarks import aggregate, score_prediction


def load_predictions(path: Path) -> dict:
    files = sorted(path.glob("predictions.rank*.jsonl")) if path.is_dir() else [path]
    predictions = {}
    for file in files:
        for line in file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["id"] in predictions:
                    raise ValueError(f"duplicate prediction {row['id']}")
                predictions[row["id"]] = row
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument("--judge", action="store_true", help="score MM-Vet / MathVista free-form with the LLM judge")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    predictions = load_predictions(Path(args.predictions))
    missing = [row["id"] for row in rows if row["id"] not in predictions]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no prediction (e.g. {missing[:3]})")
    records = []
    for row in rows:
        prediction = predictions[row["id"]]
        if prediction["status"] != "success":
            records.append(dict(id=row["id"], score=0.0, correct=False, judge_required=False, prediction="",
                                gold=str(row["answer"])))
            continue
        record = score_prediction(row["benchmark"], prediction["prediction"], row["answer"], row.get("options"),
                                  row.get("qtype", "mcq"))
        records.append(dict(record, id=row["id"]))
    pending = [i for i, r in enumerate(records) if r["judge_required"]]
    if pending and args.judge:
        from ..scoring.judge import LLMJudge
        judge = LLMJudge()

        def run(index):
            row = rows[index]
            result = judge.score(row["benchmark"], row["question"], str(row["answer"]),
                                 predictions[row["id"]]["prediction"])
            records[index].update(score=result["score"], correct=result["score"] == 1.0, judge_required=False,
                                  judge=result)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, pending))
    by_task = {}
    for row, record in zip(rows, records):
        by_task.setdefault(row["benchmark"], ([], []))
        by_task[row["benchmark"]][0].append(record)
        by_task[row["benchmark"]][1].append(row)
    summary = {}
    for task, (task_records, task_rows) in by_task.items():
        if any(r["judge_required"] for r in task_records):
            done = [r for r in task_records if not r["judge_required"]]
            summary[task] = dict(rows=len(task_records), score=None, pending_judge=len(task_records) - len(done))
            continue
        score, rule, details = aggregate(task, task_records, task_rows)
        summary[task] = dict(rows=len(task_records), score=100 * score, aggregation=rule, **details,
                             calls=sum(len(predictions[r["id"]].get("calls", [])) for r in task_rows),
                             failures=sum(predictions[r["id"]]["status"] != "success" for r in task_rows))
    print(json.dumps(summary, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(dict(summary=summary, records=records), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
