"""Datasets: loading, splitting, and synthetic generation.

Most fine-tuning effort is data, not training. This module covers the three data jobs:
1. **Load/save** examples as JSONL (a text file with one JSON object per line — the standard
   format for LLM training data because it's easy to stream, diff, and append).
2. **Split** into train/validation sets — the val set is data the model never trains on, kept
   aside so we can honestly measure whether the model *learned* or just *memorized*.
3. **Generate synthetic data** — a cheap way to bootstrap a task before you have human labels
   (with the caveat that synthetic data inherits its generator's blind spots).
"""

from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel, Field

from lora_finetune_lab.errors import DatasetError


class InstructionExample(BaseModel):
    """One supervised example: an instruction (+ optional input) → a target output.

    This is the atom of fine-tuning. Think of it as a flashcard:
    - instruction: what to do            ("Reverse the input string.")
    - input:       what to do it to      ("llama") — optional, blank for self-contained tasks
    - output:      the answer we want    ("amall")
    Show the model a few hundred flashcards and it learns the *pattern*, not just the answers.
    """

    instruction: str
    input: str = Field(default="")
    output: str


def load_jsonl(path: Path) -> list[InstructionExample]:
    """Load instruction examples from a JSONL file (one JSON object per line)."""
    examples: list[InstructionExample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue  # tolerate blank lines
        try:
            # Pydantic parses AND validates each line — a missing field or wrong type is
            # caught here, with the line number, instead of blowing up mid-training.
            examples.append(InstructionExample.model_validate_json(line))
        except ValueError as exc:
            raise DatasetError(f"Bad record on line {i}: {exc}") from exc
    return examples


def save_jsonl(examples: list[InstructionExample], path: Path) -> None:
    # Create the parent folder if needed, then write one JSON object per line.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(e.model_dump_json() for e in examples) + "\n", encoding="utf-8")


def split(
    examples: list[InstructionExample], *, val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Deterministically shuffle and split into (train, val).

    Why shuffle? So the val set isn't accidentally "all the examples from the end of the file"
    (which might all be one task type). Why a fixed seed? So everyone who runs this gets the
    exact same split — results stay comparable across machines and reruns.
    """
    shuffled = list(examples)  # copy first: never silently reorder the caller's list
    random.Random(seed).shuffle(shuffled)
    # Hold out ~10% for validation, but always at least 1 example (if we have any at all).
    n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


def generate_synthetic(n: int, *, seed: int = 42) -> list[InstructionExample]:
    """Generate `n` deterministic synthetic examples across a few task templates.

    A teaching stand-in for real data: arithmetic, casing, and reversal tasks with known answers.
    These tasks are chosen because a program can verify the answers — perfect for tests and demos.
    Real synthetic-data pipelines use a strong model to draft examples, then filter/verify them.
    """
    rng = random.Random(seed)  # seeded generator: same n + seed always yields the same data
    examples: list[InstructionExample] = []
    for _ in range(n):
        # Pick one of three task types at random, so the dataset mixes skills.
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
