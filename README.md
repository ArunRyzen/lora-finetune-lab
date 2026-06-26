<div align="center">

# 🔧 lora-finetune-lab

**QLoRA fine-tuning — done with judgment.**

A Colab-ready QLoRA notebook, CPU-testable data/prompt/eval code, and the analysis that matters most:
**when to fine-tune vs RAG vs prompt.**

</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/Arunops700/lora-finetune-lab.git && cd lora-finetune-lab
uv sync --extra dev          # CPU-only — no GPU or API keys needed
uv run lora eval             # compare base vs fine-tuned vs RAG
```
*CPU-only.* Actual training runs on a free Colab T4 — open `notebooks/qlora_finetune.ipynb`.

---

## Problem

Fine-tuning is the flashy answer and usually the *wrong* first move. Teams reach for it to add
*knowledge* (that's RAG's job) or with too little data (prompting wins). This repo treats fine-tuning
as an engineering decision: a clear **decision framework**, a **reproducible QLoRA workflow** that
runs on a *free* GPU, and an **evaluation** that proves the tune beat the cheaper alternatives — or
admits it didn't.

> **Start here:** [`docs/when-to-finetune.md`](docs/when-to-finetune.md) — the framework is the point.

## What's in here

| Piece | Runs where | Tested? |
|-------|-----------|:-------:|
| `notebooks/qlora_finetune.ipynb` | Colab/Kaggle **GPU** | manual |
| Data: build / split / **synthetic generation** (`dataset.py`) | CPU | ✅ |
| Prompt formatting (`prompts.py`) | CPU | ✅ |
| Eval: base vs tuned vs RAG (`evaluation.py`) | CPU | ✅ |
| Training step (`train.py`, QLoRA via PEFT/TRL) | GPU (`[train]` extra) | import-guarded |
| Config-as-code (`config.py`) | CPU | ✅ |

The heavy training stack (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`) is an **optional
extra**, so CI and local dev stay light; everything *around* training is plain, tested CPU Python.

## QLoRA in one paragraph

**LoRA** trains tiny low-rank adapter matrices on top of frozen weights (a few MB to ship, not a
full model copy). **QLoRA** additionally loads the frozen base in **4-bit**, cutting memory enough
that a 0.5–7B model fine-tunes on a *free* Colab **T4**. Near-full-fine-tune quality on a narrow
task, at a fraction of the cost.

## Quickstart

**Decide → data → train (Colab) → evaluate.**

```bash
# CPU: explore the data + eval workflow locally
uv sync --extra dev
uv run lora synth --n 20            # generate synthetic instruction data
uv run lora eval                    # compare base vs fine-tuned vs RAG (demo)
uv run lora config                  # print the default training config
```
```
base (untuned)     accuracy=0.00
fine-tuned         accuracy=1.00   ← learned the skill
rag baseline       accuracy=0.00   ← knowledge tool can't teach a skill
```

**Train on Colab:** open [`notebooks/qlora_finetune.ipynb`](notebooks/qlora_finetune.ipynb), set the
runtime to **T4 GPU**, and run — it generates data, trains the adapter, and evaluates base vs tuned.

## Tech stack

`Python 3.12` · `PEFT` · `TRL` · `transformers` · `bitsandbytes` (GPU) · `Pydantic v2` · `Typer` ·
`uv` · `ruff` · `mypy` · `pytest` · `GitHub Actions`

## Library

```python
from lora_finetune_lab.dataset import generate_synthetic, split
from lora_finetune_lab.prompts import format_text
from lora_finetune_lab.config import TrainingConfig
from lora_finetune_lab.train import run_training      # needs the [train] extra + a GPU

train, val = split(generate_synthetic(400))
run_training(TrainingConfig(epochs=2), [format_text(e) for e in train])
```

## Testing

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```
15 tests, **CPU-only** — data, prompts, synthetic generation, splitting, and the base-vs-tuned eval
all run with no GPU and no network. CI gates lint + types + tests.

## Future improvements
- DPO / preference tuning after SFT.
- A real-data example (not just synthetic) with a held-out benchmark.
- Adapter merging + quantized serving (vLLM) notes.
- Hook the eval into [`llm-eval-kit`](https://github.com/Arunops700/llm-eval-kit)'s gate.

## Learn more
- [`docs/when-to-finetune.md`](docs/when-to-finetune.md) — **the decision framework**
- [`docs/architecture.md`](docs/architecture.md) · [`docs/interview-questions.md`](docs/interview-questions.md) · [`docs/lessons-learned.md`](docs/lessons-learned.md)

## License
[MIT](LICENSE) · Part of my [AI_Engineer](https://github.com/Arunops700/AI_Engineer) portfolio (Milestone 5).
