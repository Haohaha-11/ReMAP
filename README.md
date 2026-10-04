# ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory

ReMAP lets a multimodal LLM look again while it reasons. During decoding the model
can emit one of two request tokens; the environment then inserts a short block of
latent visual memory into the sequence and decoding continues:

* **Global memory** (`<mem_g>`): a question-conditioned summary of every input
  image (16 latent tokens for one to three images, 4N + 4 for N images otherwise),
  computed once per question.
* **Local memory** (`<mem_l>`): region-level evidence chosen from the current
  reasoning state. A selector keeps up to three images, the Twig locator predicts a
  region in each, the regions are re-encoded at higher resolution and compressed
  into 32 latent tokens.

Memory formation is learned with supervised objectives (Stage I); when and which
memory to request is learned with reinforcement learning over branched rollouts
(Stage II).

> **Status.** This release contains the code of the latest version of ReMAP.
> Model weights and training data are not included yet.

## Repository layout

```
remap/
  modeling/
    model.py          ReMAP model: base Qwen2.5-VL, memory tokens, Global/Local blocks
    global_memory.py  Image/Relation weaver, clause writer, bridge, prefix readout
    local_memory.py   source selector, Local weaver, global conditioner, set context
    relation.py       relation memory that conditions Local requests
    twig.py           Twig region locator
    fact_writer.py    structured fact writer that provides Global clauses (1-3 images)
    memory_tokens.py  memory request / end tokens and their trainable rows
  inference/
    session.py        decoding with memory requests (evaluation and RL rollouts)
tests/
```

## Installation

```bash
pip install -e .            # PyTorch 2.4, Transformers 4.51.3, PEFT 0.19.1
pip install flash-attn==2.7.3 --no-build-isolation
```

ReMAP builds on the Q-Zoom Qwen2.5-VL-7B checkpoint
([YuhengSSS/Q-Zoom-Qwen2.5VL-7B](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-7B)).

## Citation

```bibtex
@article{remap2026,
  title  = {ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory},
  author = {Jiang, Hao and Guo, Zhanyu and others},
  year   = {2026}
}
```

## License

Apache License 2.0. See [NOTICE](NOTICE) for third-party attributions.
