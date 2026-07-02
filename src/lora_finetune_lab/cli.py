"""Command-line interface for the data/eval workflow (training runs in the notebook).

This is the "front door" of the repo. Each function below becomes a `lora <name>` subcommand
thanks to Typer (a library that turns plain Python functions into CLI commands). Nothing here
does the heavy GPU work — the CLI is for the cheap, local parts: making data, inspecting it,
printing config, and running the base-vs-fine-tuned-vs-RAG comparison demo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from lora_finetune_lab.config import TrainingConfig
from lora_finetune_lab.dataset import generate_synthetic, load_jsonl, save_jsonl, split
from lora_finetune_lab.errors import GeminiDependencyError
from lora_finetune_lab.evaluation import ConstantModel, Model, RuleModel, compare
from lora_finetune_lab.prompts import format_text

app = typer.Typer(
    help="QLoRA fine-tuning lab: data, eval, and config tooling.", no_args_is_help=True
)


@app.command()
def synth(
    n: Annotated[int, typer.Option(help="Number of examples.")] = 20,
    out: Annotated[Path | None, typer.Option(help="Write JSONL here.")] = None,
) -> None:
    """Generate a synthetic instruction dataset."""
    examples = generate_synthetic(n)
    if out:
        # --out given: save everything to a JSONL file (one example per line) for training later.
        save_jsonl(examples, out)
        typer.echo(f"Wrote {len(examples)} examples to {out}", err=True)
    else:
        # No --out: just preview a few examples on screen so you can see what the data looks like.
        for ex in examples[:5]:
            typer.echo(f"{ex.instruction!r} (in={ex.input!r}) -> {ex.output!r}")
        if n > 5:
            typer.echo(f"... and {n - 5} more (use --out to save all)", err=True)


@app.command()
def inspect(data: Annotated[Path, typer.Option(help="Dataset JSONL.")]) -> None:
    """Show dataset stats and one formatted training example."""
    examples = load_jsonl(data)
    train, val = split(examples)
    typer.echo(f"total={len(examples)} train={len(train)} val={len(val)}")
    if examples:
        # Show exactly what one example looks like AFTER prompt formatting — i.e. the literal
        # text the model would be trained on. Seeing this catches formatting mistakes early.
        typer.echo("\n--- formatted training example ---")
        typer.echo(format_text(examples[0]))


@app.command(name="eval")
def evaluate() -> None:
    """Compare base vs fine-tuned vs RAG on a held-out synthetic set (demo)."""
    # Make 40 examples, then keep only the held-out slice (data the "models" never trained on).
    # Scoring on training data would be like grading students on questions they've already seen.
    _, held_out = split(generate_synthetic(40))

    # >>> THIS is the base-vs-fine-tuned-vs-RAG comparison the whole repo builds toward. <<<
    # Each entry is anything with a `.generate(prompt) -> str` method (the Model protocol), so
    # stand-ins and real models are interchangeable. The demo casting:
    #   - "base (untuned)": always answers "I'm not sure" — a model that never learned the skill.
    #   - "fine-tuned":     actually solves the tasks — a model that DID learn the skill.
    #   - "rag baseline":   RAG retrieves documents (knowledge), but these tasks need a SKILL
    #     (adding numbers, reversing strings), so retrieval finds nothing useful. This is the
    #     "RAG adds knowledge, fine-tuning adds behaviour" lesson from docs/when-to-finetune.md.
    models: dict[str, Model] = {
        "base (untuned)": ConstantModel("I'm not sure."),
        "fine-tuned": RuleModel(),
        "rag baseline": ConstantModel("Retrieved docs didn't contain the answer."),
    }

    # Optional 4th contender: a real hosted model (Gemini) to answer "would a strong general
    # model, just prompted, already solve this?". Only joins if you export GEMINI_API_KEY —
    # without a key this command stays fully offline. See src/lora_finetune_lab/gemini.py.
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from lora_finetune_lab.gemini import GeminiModel

            models["gemini (api)"] = GeminiModel()
        except GeminiDependencyError as exc:
            typer.echo(f"(Gemini skipped: {exc})", err=True)

    # `compare` runs every model over the same held-out examples and returns name -> accuracy.
    for name, acc in compare(models, held_out).items():
        typer.echo(f"{name:18s} accuracy={acc:.2f}  (n={len(held_out)})")
    typer.echo(
        "\nLesson: fine-tuning teaches the skill; RAG can't (no knowledge gap here).", err=True
    )


@app.command()
def config() -> None:
    """Print the default training configuration as JSON."""
    typer.echo(TrainingConfig().model_dump_json(indent=2))


@app.command()
def train(
    data: Annotated[Path, typer.Option(help="Training dataset JSONL.")],
    out: Annotated[str, typer.Option(help="Adapter output dir.")] = "outputs/adapter",
) -> None:
    """Run QLoRA training (requires a GPU + the `train` extra; otherwise use the notebook)."""
    # Imported here (not at the top) so `lora synth`/`lora eval` work on machines without
    # the GPU stack installed — the import only has to succeed if you actually train.
    from lora_finetune_lab.errors import TrainingDependencyError
    from lora_finetune_lab.train import run_training

    examples = load_jsonl(data)
    texts = [format_text(ex) for ex in examples]
    cfg = TrainingConfig(output_dir=out)
    try:
        path = run_training(cfg, texts)
        typer.echo(f"Saved adapter to {path}")
    except TrainingDependencyError as exc:
        # No torch/peft installed: explain instead of crashing with a raw ImportError.
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
