# Data formats

All files are JSON lines unless noted.  Image paths may be absolute or relative to the
working directory.  No data is distributed with this release.

## Benchmark questions (`remap.evaluation`)

```json
{"id": "vstar:0", "benchmark": "vstar", "images": ["..."], "question": "<original benchmark prompt>",
 "answer": "A", "options": {"A": "rubber", "B": "cotton"}, "qtype": "mcq"}
```

`question` is the benchmark's own prompt; images are placed at its `<image>`
placeholders (all images first when it has none).  The prompt's "answer directly"
instruction is replaced by the reasoning protocol
(`remap.scoring.benchmarks.project_question`).  Grouped benchmarks also need their
group field: `source` (CV-Bench 2D), `sub_task` (BLINK), `cycle_category` (HR-Bench).
`benchmark` is one of `vstar, cvbench_2d, cvbench_3d, blink, muirbench, mmsi,
realworldqa, hrbench4k, hrbench8k, mimic_common, mimic_counting, mimic_oddone,
mimic_listing, mathvista, mmvet`.

## Stage II questions (`remap.training.rl.train`)

```json
{"id": "muirbench:00001815", "benchmark": "muirbench", "images": ["..."],
 "question": "<prompt shown to the policy, ending with the Answer: protocol>",
 "max_new_tokens": 768,
 "answer": {"backend": "typed", "answer_type": "public_mcq", "gold_answer": "A",
            "acceptable_answers": ["A"], "public_options": {"A": "...", "B": "..."},
            "correct_letter": "A"}}
```

In Stage II all images come first, followed by the question text.  `answer.backend`:

* `typed`: `answer_type` is one of `integer, numeric, class_set, class_name,
  short_text, option_text, public_mcq, image_index`, with `gold_answer`, optional
  `acceptable_answers`, and `public_options` + `correct_letter` (multiple choice) or
  `answer_index` (zero-based, image index) as needed;
* `mimic`: `{"backend": "mimic", "gold": "..."}`; `benchmark` names the MIMIC task;
* `judge`: `{"backend": "judge", "gold": "...", "question": "<benchmark prompt>"}`,
  scored by the LLM judge (`remap.scoring.judge`).

## Stage I states (`remap.training.stage1`)

`stage1.collect` writes one `torch.save` file per memory request
(`remap.training.stage1.states.MemoryState`) and a line per state:

```json
{"state_id": "...", "row_id": "...", "kind": "local", "state": "states/.../x_c0.pt",
 "question": "...", "images": ["..."], "prefix_text": "<student text before the request>",
 "previous_reads": [{"kind": "global", "views": [{"path": "...", "image_index": 0}]}],
 "current_read": {"kind": "local", "views": [{"path": "<crop>.png", "image_index": 1}]}}
```

`stage1.annotate` (continuations) and `stage1.review_roi` (Twig boxes) write one
`target.json` per state and a training manifest of approved states:

```json
{"state": "states/.../x_c0.pt", "target": "annotations/x_c0/target.json", "kind": "local"}
```

A continuation target holds `suffix_text` and `endpoint` (`final_answer` adds EOS to
the targets, `partial` does not); a Twig target holds
`targets: [{"image_index": 1, "bbox_normalized": [x0, y0, x1, y1]}]`.

## Trajectories (`remap.training.interface_warmup`, `remap.training.policy_sft`)

```json
{"id": "...", "images": ["..."], "question": "...", "token_ids": [..., 151645],
 "loss_mask": [true, false, ...]}
```

`token_ids` is a complete generated sequence including memory requests and the final
EOS; images come first in the prompt.  The warm-up uses every token as a target;
policy SFT uses `loss_mask`.
