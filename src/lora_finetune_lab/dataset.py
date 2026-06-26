"""Datasets: loading, splitting, and synthetic generation.

Most fine-tuning effort is data, not training. This module builds instruction examples, splits them
deterministically, and can **generate synthetic data** — a cheap way to bootstrap a task before you
have human labels (with the caveat that synthetic data inherits the generator's blind spots).
"""

from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel, Field

from lora_finetune_lab.errors import DatasetError


class InstructionExample(BaseModel):
    """One supervised example: an instruction (+ optional input) → a target output."""

    instruction: str
    input: str = Field(default="")
    output: str


def load_jsonl(path: Path) -> list[InstructionExample]:
    """Load instruction examples from a JSONL file (one JSON object per line)."""
    examples: list[InstructionExample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(InstructionExample.model_validate_json(line))
        except ValueError as exc:
            raise DatasetError(f"Bad record on line {i}: {exc}") from exc
    return examples


def save_jsonl(examples: list[InstructionExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(e.model_dump_json() for e in examples) + "\n", encoding="utf-8")


def split(
    examples: list[InstructionExample], *, val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Deterministically shuffle and split into (train, val)."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


def generate_synthetic(n: int, *, seed: int = 42) -> list[InstructionExample]:
    """Generate `n` deterministic synthetic examples across a few task templates.

    A teaching stand-in for real data: arithmetic, casing, and reversal tasks with known answers.
    Real synthetic-data pipelines use a strong model to draft examples, then filter/verify them.
    """
    rng = random.Random(seed)
    examples: list[InstructionExample] = []
    for _ in range(n):
        kind = rng.choice(["add", "upper", "reverse"])
        if kind == "add":
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            examples.append(
                InstructionExample(instruction=f"What is {a} + {b}?", output=str(a + b))
            )
        elif kind == "upper":
            word = rng.choice(["model", "vector", "agent", "token", "prompt"])
            examples.append(
                InstructionExample(
                    instruction="Convert the input to uppercase.", input=word, output=word.upper()
                )
            )
        else:
            word = rng.choice(["llama", "qlora", "adapter", "tensor"])
            examples.append(
                InstructionExample(
                    instruction="Reverse the input string.", input=word, output=word[::-1]
                )
            )
    return examples
