# ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory

[![arXiv](https://img.shields.io/badge/arXiv-2610.05097-b31b1b.svg)](https://arxiv.org/abs/2610.05097)

Hao Jiang\*, Zhanyu Guo\*, Chenwei Wu, Yichen Guo, Qizhe Zhang, Junchi Yao, Jixian Wu,
Jinhao You, Kai Tang, Jiajun Cao, Tinghao Wang, Mengyu Wang, Leo Anthony Celi,
Shanghang Zhang (\* equal contribution)

ReMAP lets a multimodal LLM look again while it reasons. During decoding the model
can emit one of two request tokens; a block of latent visual memory is then inserted
into the sequence and decoding continues:

* **Global memory** (`<mem_g>`): a static, question-conditioned summary of all input
  images and their relations (16 latent tokens for one to three images, 4N + 4 for N
  images otherwise).
* **Local memory** (`<mem_l>`): region-level evidence chosen for the current reasoning
  state. Up to three images are selected, the Twig locator predicts a region in each,
  and the regions are re-encoded at higher resolution and compressed into 32 latent
  tokens, conditioned on the relation memory of the images.

Memory formation is learned with supervised objectives (Stage I); when to request
which memory is learned with reinforcement learning over branched rollouts (Stage II).

> **Status (v1).** This release contains the inference, training and evaluation code of
> ReMAP. Model weights and training data are not released yet. The code is a cleaned-up
> version of our research code; it was checked to produce the same outputs as the
> original implementation on test cases.

## Installation

```bash
git clone <this repository> && cd <repository>
pip install -e .                       # PyTorch 2.4, Transformers 4.51.3, PEFT 0.19.1
pip install flash-attn==2.7.3 --no-build-isolation
pip install -e ".[data,judge]"         # optional: teacher API client, LLM judge
```

ReMAP builds on the Q-Zoom Qwen2.5-VL-7B checkpoint
([YuhengSSS/Q-Zoom-Qwen2.5VL-7B](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-7B)).
A ReMAP checkpoint directory holds `remap_config.json`, `memory.safetensors` (memory
modules, Twig, memory-token rows and the policy LoRA), `fact_writer.safetensors`,
`fact_writer_config.json` and `chat_template.json`.

## Inference

```python
from remap.modeling import FactWriter, ReMAP
from remap.inference.session import DecodeConfig, generate, prompt_content
from remap.scoring.benchmarks import project_question

base = "YuhengSSS/Q-Zoom-Qwen2.5VL-7B"
model = ReMAP.from_pretrained("checkpoints/remap", base_model=base)
writer = FactWriter.from_pretrained("checkpoints/remap", base_model=base)   # Global clauses, 1-3 images

question = project_question("Question: <image> <image> Which image shows ...?\nChoices:\n(A) ...")
images = ["a.jpg", "b.jpg"]
result = generate(model, images, prompt_content(question, len(images)), question=question,
                  fact_writer=writer, config=DecodeConfig(max_new_tokens=768, max_calls=4))
print(result.text)                         # reasoning, ending with "Answer: ..."
print([event.kind for event in result.events])   # memory requests, e.g. ['local', 'global']
```

Decoding is greedy by default. `DecodeConfig(allow_memory=False)` disables memory
requests (the "memory off" setting).

## Evaluation

```bash
torchrun --nproc_per_node 8 -m remap.evaluation.generate \
    --checkpoint checkpoints/remap --base-model YuhengSSS/Q-Zoom-Qwen2.5VL-7B \
    --data benchmarks/muirbench.jsonl --output predictions/muirbench
python -m remap.evaluation.score --data benchmarks/muirbench.jsonl --predictions predictions/muirbench
```

Benchmark files are JSON lines with the benchmark's original prompt and answer
([docs/data.md](docs/data.md)). Multiple-choice benchmarks use their official answer
parsers, MIMIC tasks the released MIMIC metrics; MM-Vet and free-form MathVista
answers are scored with an LLM judge (`--judge`, configured through `OPENAI_API_KEY`,
`OPENAI_API_URL` and `JUDGE_MODEL`).

## Training

Each stage runs with `torchrun` (one process per GPU) and a YAML config in
[configs/](configs); data formats are described in [docs/data.md](docs/data.md).

| Step | Module | What it trains |
|---|---|---|
| Policy SFT | `remap.training.policy_sft` | policy LoRA and memory-token rows on teacher trajectories (`configs/policy_sft.yaml`) |
| Fact writer | `remap.training.pretrain.fact_writer` | decoder and vision LoRA of the fact writer that provides Global clauses |
| Stage I data | `remap.training.stage1.collect` | records the student's state at each memory request |
| | `remap.training.stage1.annotate` | teacher continuations after each read, independently reviewed |
| | `remap.training.stage1.review_roi` | teacher-reviewed regions for the student's Local reads |
| Stage I | `remap.training.stage1.train` | Global memory, Local weaver, or Twig (`configs/stage1_{global,local,twig}.yaml`) |
| Interface warm-up | `remap.training.interface_warmup` | memory-token rows on successful complete trajectories (`configs/interface_warmup.yaml`) |
| Stage II | `remap.training.rl.train` | when to request which memory: policy LoRA and request-token rows (`configs/stage2_rl.yaml`) |

```bash
torchrun --nproc_per_node 8 -m remap.training.stage1.collect --checkpoint ckpt --base-model <base> \
    --questions train.jsonl --output states/
cat states/states.rank*.jsonl > states/states.jsonl
python -m remap.training.stage1.annotate --states states/states.jsonl --questions train.jsonl \
    --output annotations/                         # TEACHER_API_URL, TEACHER_API_KEY
torchrun --nproc_per_node 8 -m remap.training.stage1.train --config configs/stage1_local.yaml
torchrun --nproc_per_node 8 -m remap.training.rl.train --config configs/stage2_rl.yaml
```

The teacher is any OpenAI-compatible multimodal chat endpoint; we used
Qwen3.5-397B-A17B. Stage II rewards combine the task score with a bounded, decaying
bonus for memory reads (`remap/training/rl/rewards.py`).

The relation producer, the Local image selector, set-context fusion and answer reader,
and the Global-to-Local conditioner were trained in earlier experiments with model
variants that this release does not contain; they are kept frozen in all stages above,
and their training code is not included.

## Repository layout

```
remap/
  modeling/      ReMAP model: memory tokens, Global / Local memory, relation memory, Twig, fact writer
  inference/     decoding with memory requests (evaluation and RL rollouts)
  scoring/       typed answer verification, benchmark parsers and metrics, LLM judge
  evaluation/    batch generation and scoring
  data/          teacher API client
  training/      stage1/, rl/, interface warm-up, policy SFT, fact writer
configs/         one YAML file per training stage
docs/data.md     data formats
tests/           CPU unit tests
```

## Citation

```bibtex
@article{jiang2026remap,
  title   = {ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory},
  author  = {Jiang, Hao and Guo, Zhanyu and Wu, Chenwei and Guo, Yichen and Zhang, Qizhe and
             Yao, Junchi and Wu, Jixian and You, Jinhao and Tang, Kai and Cao, Jiajun and
             Wang, Tinghao and Wang, Mengyu and Celi, Leo Anthony and Zhang, Shanghang},
  journal = {arXiv preprint arXiv:2610.05097},
  year    = {2026}
}
```

## License

Apache License 2.0. See [NOTICE](NOTICE) for third-party attributions.
