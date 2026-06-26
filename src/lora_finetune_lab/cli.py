"""Command-line interface for the data/eval workflow (training runs in the notebook)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lora_finetune_lab.config import TrainingConfig
from lora_finetune_lab.dataset import generate_synthetic, load_jsonl, save_jsonl, split
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
        save_jsonl(examples, out)
        typer.echo(f"Wrote {len(examples)} examples to {out}", err=True)
    else:
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
        typer.echo("\n--- formatted training example ---")
        typer.echo(format_text(examples[0]))


@app.command(name="eval")
def evaluate() -> None:
    """Compare base vs fine-tuned vs RAG on a held-out synthetic set (demo)."""
    _, held_out = split(generate_synthetic(40))
    models: dict[str, Model] = {
        "base (untuned)": ConstantModel("I'm not sure."),
        "fine-tuned": RuleModel(),
        "rag baseline": ConstantModel("Retrieved docs didn't contain the answer."),
    }
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
    from lora_finetune_lab.errors import TrainingDependencyError
    from lora_finetune_lab.train import run_training

    examples = load_jsonl(data)
    texts = [format_text(ex) for ex in examples]
    cfg = TrainingConfig(output_dir=out)
    try:
        path = run_training(cfg, texts)
        typer.echo(f"Saved adapter to {path}")
    except TrainingDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
