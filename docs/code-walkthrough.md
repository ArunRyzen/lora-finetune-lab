# Code Walkthrough (for beginners)

A plain-English tour of every file in this repo: what it does, why it exists, and where to look
when you want to change something. No fine-tuning experience assumed.

## The 30-second big picture

This repo teaches one workflow: **decide → data → train → evaluate.**

1. **Decide** whether fine-tuning is even the right tool ([`docs/when-to-finetune.md`](when-to-finetune.md)).
2. **Build data**: small "flashcard" examples of the behaviour you want (`dataset.py`, `prompts.py`).
3. **Train** tiny LoRA adapters on a free Colab GPU (`train.py`, the notebook).
4. **Evaluate** honestly: did the fine-tune beat the untouched model *and* the cheaper
   alternatives? (`evaluation.py`, `lora eval`).

Everything except step 3 runs on your laptop with no GPU and no API key.

## First: what are LoRA and QLoRA, in plain words?

A language model is millions/billions of numbers ("weights"). *Full* fine-tuning changes all of
them — that needs serious GPUs and produces a giant copy of the model.

- **LoRA** says: *don't touch the original model at all*. Freeze it, and bolt a few **tiny
  trainable add-on layers** (called **adapters**) onto it. During training, only the adapters
  change. The result you save and ship is just the adapters — a few **megabytes**, like a
  lens screwed onto a camera body rather than a whole new camera.
- **QLoRA** adds one more memory trick: since the big frozen model is only *read* during
  training (never updated), store it in compressed **4-bit** form. Memory use drops ~4x.

Together, these two tricks are why a **free Colab T4 GPU** is enough to fine-tune a real model —
which is the entire premise of this repo.

## Reading order

Read the files in this order — each one builds on the previous:

| # | File | Read it to learn... |
|---|------|---------------------|
| 1 | [`docs/when-to-finetune.md`](when-to-finetune.md) | *whether* to fine-tune at all (the framework everything else serves) |
| 2 | [`src/lora_finetune_lab/dataset.py`](../src/lora_finetune_lab/dataset.py) | what one training example looks like; loading/splitting/synthetic data |
| 3 | [`src/lora_finetune_lab/prompts.py`](../src/lora_finetune_lab/prompts.py) | how an example becomes the exact text the model sees |
| 4 | [`src/lora_finetune_lab/evaluation.py`](../src/lora_finetune_lab/evaluation.py) | the `Model` protocol and how models are scored and compared |
| 5 | [`src/lora_finetune_lab/cli.py`](../src/lora_finetune_lab/cli.py) | the `lora` commands — especially `lora eval` |
| 6 | [`src/lora_finetune_lab/config.py`](../src/lora_finetune_lab/config.py) | every training knob, explained one by one |
| 7 | [`src/lora_finetune_lab/train.py`](../src/lora_finetune_lab/train.py) | the actual QLoRA training step (GPU only) |
| 8 | [`notebooks/qlora_finetune.ipynb`](../notebooks/qlora_finetune.ipynb) | all of it end-to-end on a free Colab GPU |

The other files: `errors.py` (the repo's own exception types), `gemini.py` (optional API model
for `lora eval`, see below), `tests/` (offline checks for all of the above), `pyproject.toml`
(dependencies — note how the GPU stack is an *optional* extra so normal installs stay light).

## File-by-file tour

### `dataset.py` — the flashcards

The atom of fine-tuning is `InstructionExample`:

```python
class InstructionExample(BaseModel):
    instruction: str          # what to do:        "Reverse the input string."
    input: str = Field(...)   # what to do it to:  "llama"   (optional)
    output: str               # the answer we want: "amall"
```

It's a Pydantic `BaseModel`, which means every example is *validated* — load a JSONL file with a
missing field and `load_jsonl` tells you the exact bad line instead of crashing mid-training.

Two other functions matter:

- `split(examples)` — shuffles (with a fixed seed, so it's repeatable) and holds out ~10% as a
  **validation set**. That held-out slice is the "exam paper" the model never studies from;
  every honest score in this repo is measured on it.
- `generate_synthetic(n)` — manufactures practice tasks (adding numbers, uppercasing, reversing
  strings). They're deliberately toy tasks *whose answers a program can verify*, which makes the
  whole pipeline testable without a GPU. Real projects would use human data or model-drafted,
  human-verified synthetic data.

### `prompts.py` — from flashcard to model-ready text

Models are trained on one long string, with markers showing who is speaking. `format_text` builds it:

```python
parts = [
    f"<|system|>\n{SYSTEM_PROMPT}",                              # standing instructions
    f"<|user|>\n{_user_turn(example)}",                          # instruction (+ input)
    "<|assistant|>\n" + (example.output if include_output else ""),  # the answer... or not
]
```

Line by line:

- `<|system|>` — a fixed "be helpful and concise" instruction, identical for every example.
- `<|user|>` — the instruction, plus the input (if any) after a blank line.
- `<|assistant|>` — the crucial switch. With `include_output=True` (**training**) the answer is
  included, so the model learns to produce it. With `include_output=False` (**evaluation**) the
  text stops right after `<|assistant|>\n` — an open bracket the model must fill in itself.

The one rule this module enforces: **format examples identically for training and inference.**
A mismatch (training with markers, prompting without) is the most common reason a fine-tuned
model mysteriously underperforms.

### `evaluation.py` — the honest exam ⭐

This is the most important file in the repo. Start with the `Model` protocol:

```python
class Model(Protocol):
    def generate(self, prompt: str) -> str: ...
```

Read it as: *"anything with a `generate(prompt) -> str` method counts as a Model."* No
inheritance, no registration — Python's `Protocol` checks the shape, not the family tree.
That's why four very different things all plug into the same scoring code:

- the rule-based fakes below (offline demo + tests),
- a real Hugging Face model in the Colab notebook (the `HFModel` wrapper in step 4),
- the optional Gemini API adapter in `gemini.py`,
- anything you write next.

`accuracy(model, examples)` runs the exam: for each example it formats the prompt *without* the
answer, asks the model, and scores a point if the expected answer appears anywhere in the reply
(lenient on purpose — "The answer is 42." counts for "42"). `compare(models, examples)` just
runs `accuracy` for several named models **on the same questions**, returning `{name: score}`.

The two demo models are the punchline:

- `RuleModel` solves the tasks with if/else rules — it *plays the role of* a successfully
  fine-tuned model (one that learned the skill).
- `ConstantModel` always gives the same canned reply — it plays both the untuned base model
  ("I'm not sure.") and the RAG baseline ("Retrieved docs didn't contain the answer.").

Why is a canned reply a fair stand-in for RAG here? Because RAG's whole move is *looking up
documents and pasting them into the prompt* — it adds **knowledge**. These tasks have no missing
knowledge; they need a **skill** (add, reverse, uppercase). Retrieval comes back empty-handed
every time, so a constant "found nothing" reply models it perfectly.

### `cli.py` — the `lora` command, and WHERE the comparison happens

`cli.py` uses Typer to turn functions into subcommands: `lora synth`, `lora inspect`,
`lora eval`, `lora config`, `lora train`.

**The base-vs-fine-tuned-vs-RAG comparison lives in the `evaluate` function in
[`src/lora_finetune_lab/cli.py`](../src/lora_finetune_lab/cli.py)** (the `@app.command(name="eval")`
one). It's short enough to read whole:

```python
_, held_out = split(generate_synthetic(40))          # keep only the unseen "exam" slice
models: dict[str, Model] = {
    "base (untuned)": ConstantModel("I'm not sure."),
    "fine-tuned": RuleModel(),
    "rag baseline": ConstantModel("Retrieved docs didn't contain the answer."),
}
for name, acc in compare(models, held_out).items():  # same exam for every contender
    typer.echo(f"{name:18s} accuracy={acc:.2f}  (n={len(held_out)})")
```

Run `uv run lora eval` and you get:

```
base (untuned)     accuracy=0.00
fine-tuned         accuracy=1.00
rag baseline       accuracy=0.00
```

That table **is the repo's thesis in three lines**: the fine-tuned model wins because the gap
was *behaviour* (a skill to learn), and **RAG scores zero because RAG adds knowledge, not
behaviour** — pointing a search engine at an arithmetic problem doesn't teach anyone to add.
Flip the scenario (ask about yesterday's news) and the ranking flips too: fine-tuning can't
inject fresh facts, retrieval can. Which tool fixes which gap is the whole decision framework —
read the "one-line rule" and "Do NOT fine-tune when…" sections of
[`docs/when-to-finetune.md`](when-to-finetune.md) with this output in front of you.

### `config.py` — every training knob in one place

A single Pydantic class, `TrainingConfig`, holds all hyperparameters: which base model, the LoRA
adapter size (`lora_r`), how strongly adapters mix in (`lora_alpha`), the 4-bit switch
(`load_in_4bit`), learning rate, epochs, batch sizes. Two reasons it's a class and not scattered
constants: values are **validated** on creation (a typo fails instantly, not an hour into a GPU
run), and a config can be **printed/saved** (`lora config`) so every experiment is reproducible.
Each field has a plain-English comment in the file — read them there.

### `train.py` — the GPU part (safe to read, needs a GPU to run)

`run_training` is the recipe in three steps, matching the QLoRA explanation above:
**(1)** load the frozen base model in 4-bit, **(2)** attach LoRA adapters to its attention
layers, **(3)** run supervised fine-tuning and save *just the adapter* (a few MB).

Note the `try/except ImportError` at the top: the heavy libraries (`torch`, `peft`, `trl`…) are
imported *inside* the function, so simply importing this package never requires them. On a
laptop without the `[train]` extra you get a clear `TrainingDependencyError` telling you to use
the Colab notebook — not a cryptic crash. The same "optional dependency" pattern is used by
`gemini.py`.

### `gemini.py` — an optional reality check (API)

Entirely optional, and off by default: if you `pip install '.[gemini]'` and export
`GEMINI_API_KEY`, then `lora eval` adds a fourth contender — Google's hosted `gemini-2.5-flash`
model, called live over the API. Why include an API model in a local-fine-tuning repo? Because
the honest pre-question to any fine-tune is *"would a strong general model, just prompted,
already do this?"* If a hosted model aces your task with zero training, prompting is probably
the cheaper answer (rung 1 of the ladder in `when-to-finetune.md`). Without a key, nothing
changes — the repo stays 100% offline.

### `notebooks/` — everything, end to end (two flavours)

Both notebooks glue it all together on a real free GPU (Colab or Kaggle — see
[`free-gpu-guide.md`](free-gpu-guide.md)): install → generate data with `dataset.py` → format
with `prompts.py` → train → score base vs tuned with `evaluation.accuracy`. Each stage has its
own markdown explainer inside the notebook.

- **`qlora_finetune.ipynb` (start here)** — the classic Hugging Face stack (`peft`/`trl`/
  `bitsandbytes`) on a 0.5B model, via this repo's own `train.py`. Best for *learning the
  mechanics* — this is the stack interview questions are about.
- **`unsloth_finetune.ipynb` (do this second)** — the same lesson powered by
  [Unsloth](https://unsloth.ai), the popular optimized trainer (~2× faster, far less GPU
  memory). The savings let it fine-tune a **6× bigger model (Llama 3.2 3B)** on the same free
  T4. It also grades the model *before and after* training on the same object — safe because
  fresh LoRA adapters start as zeros, so before training the model behaves exactly like the base.

### `tests/` — proof it works, offline

Every CPU module has tests (`uv run pytest`, no GPU, no network). Worth reading as *executable
documentation* — e.g. `test_evaluation.py::test_compare_ranks_finetuned_above_baselines` asserts
the repo's central claim, and `test_gemini.py` shows how to fake an SDK so an API adapter can be
tested offline.

## Where to find X

| I want to see... | Look in |
|---|---|
| What one training example looks like | `InstructionExample` in [`dataset.py`](../src/lora_finetune_lab/dataset.py) |
| How synthetic data is generated | `generate_synthetic` in [`dataset.py`](../src/lora_finetune_lab/dataset.py) |
| The train/validation split (and why it's seeded) | `split` in [`dataset.py`](../src/lora_finetune_lab/dataset.py) |
| The exact text the model trains on | `format_text` in [`prompts.py`](../src/lora_finetune_lab/prompts.py); or run `uv run lora inspect --data <file>` |
| The `Model` protocol (the plug-in interface) | top of [`evaluation.py`](../src/lora_finetune_lab/evaluation.py) |
| How a model's score is computed | `accuracy` in [`evaluation.py`](../src/lora_finetune_lab/evaluation.py) |
| **The base vs fine-tuned vs RAG comparison** | `evaluate` in [`cli.py`](../src/lora_finetune_lab/cli.py) → calls `compare` in [`evaluation.py`](../src/lora_finetune_lab/evaluation.py) |
| Why RAG loses here (and when it would win) | `ConstantModel` docstring in [`evaluation.py`](../src/lora_finetune_lab/evaluation.py) + [`when-to-finetune.md`](when-to-finetune.md) |
| Every training hyperparameter, explained | [`config.py`](../src/lora_finetune_lab/config.py) |
| The actual QLoRA training code | `run_training` in [`train.py`](../src/lora_finetune_lab/train.py) |
| How heavy/optional deps are import-guarded | `try/except ImportError` in [`train.py`](../src/lora_finetune_lab/train.py) and [`gemini.py`](../src/lora_finetune_lab/gemini.py) |
| The optional Gemini comparison | [`gemini.py`](../src/lora_finetune_lab/gemini.py) + the `GEMINI_API_KEY` block in [`cli.py`](../src/lora_finetune_lab/cli.py) |
| The exact prompts/replies during `lora eval` (set `LLM_DEBUG=1`) | [`debuglog.py`](../src/lora_finetune_lab/debuglog.py) + the `generate` methods in [`evaluation.py`](../src/lora_finetune_lab/evaluation.py) and [`gemini.py`](../src/lora_finetune_lab/gemini.py) |
| Whether to fine-tune at all | [`when-to-finetune.md`](when-to-finetune.md) |
| Real-GPU training, end to end (classic stack) | [`notebooks/qlora_finetune.ipynb`](../notebooks/qlora_finetune.ipynb) |
| Real-GPU training with Unsloth (bigger model, faster) | [`notebooks/unsloth_finetune.ipynb`](../notebooks/unsloth_finetune.ipynb) |
| Where to get a free GPU (Colab vs Kaggle, step by step) | [`free-gpu-guide.md`](free-gpu-guide.md) |

## The takeaway to carry with you

> **RAG adds knowledge. Fine-tuning adds behaviour.**

`lora eval` shows the behaviour side winning (a skill was missing, so the fine-tune wins and RAG
scores zero). Don't over-generalize from it: on a *knowledge* gap the table flips. Before any
fine-tune, climb the ladder in [`when-to-finetune.md`](when-to-finetune.md) — prompt first, RAG
second, fine-tune last — and demand a held-out-set win before shipping.
