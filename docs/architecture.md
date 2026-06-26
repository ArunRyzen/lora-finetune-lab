# Architecture & Design Decisions

## The split that makes this repo work
Training needs a GPU and a heavy stack; everything *around* training doesn't. So the package is
deliberately layered:

- **CPU core** (tested in CI): `config`, `prompts`, `dataset` (incl. synthetic generation),
  `evaluation`. Pure Python + Pydantic — no torch.
- **GPU edge** (`train` extra, import-guarded): `train.run_training` imports `transformers` / `peft`
  / `trl` / `bitsandbytes` *inside the function*, so importing the package never requires them.

Result: fast, green CI on the parts that hold the logic; the actual fine-tune runs on Colab.

## Key decisions

### 1. Fine-tuning is a *decision*, so the decision is the headline
The most reusable artifact is `docs/when-to-finetune.md`, not the trainer. The code exists to support
the judgment: cheap synthetic data, consistent formatting, and a base-vs-tuned-vs-RAG comparison.

### 2. Config-as-code
Every hyperparameter lives in a typed, serializable `TrainingConfig`, so a run is reproducible and
diffable — and the CLI can print it. No magic constants buried in the trainer.

### 3. One formatting function for train and inference
`prompts.format_text` formats examples identically for training and inference. A mismatch there is
the most common cause of a tuned model underperforming, so it's a single source of truth (and tested).

### 4. Evaluation behind a `Model` protocol
`evaluation` scores anything with `generate(prompt) -> str`. That's a real HF model on Colab or a
fake (`RuleModel`, `ConstantModel`) in CI — so the comparison logic is tested without a GPU, and the
demo (`lora eval`) shows the fine-tune-vs-RAG lesson concretely.

### 5. Synthetic data as a first-class helper
`generate_synthetic` bootstraps a task deterministically. It's a teaching stand-in for the real
pattern (a strong model drafts examples, then you filter/verify) — and it's documented as such, with
its blind-spot caveat.

## Trade-offs left open
- Real-data benchmark beyond synthetic tasks.
- DPO/preference tuning after SFT; adapter merging; quantized (vLLM) serving.
- Wiring the held-out eval into `llm-eval-kit`'s CI gate.
