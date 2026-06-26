"""Dataset: deterministic synth, deterministic split, JSONL round-trip, error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from lora_finetune_lab.dataset import (
    InstructionExample,
    generate_synthetic,
    load_jsonl,
    save_jsonl,
    split,
)
from lora_finetune_lab.errors import DatasetError


def test_synthetic_is_deterministic() -> None:
    a = generate_synthetic(10, seed=1)
    b = generate_synthetic(10, seed=1)
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b]
    assert len(a) == 10


def test_synthetic_answers_are_correct() -> None:
    for ex in generate_synthetic(30):
        if ex.instruction.startswith("Reverse"):
            assert ex.output == ex.input[::-1]
        elif ex.instruction.startswith("Convert"):
            assert ex.output == ex.input.upper()


def test_split_is_deterministic_and_partitions() -> None:
    examples = generate_synthetic(20)
    train, val = split(examples, val_fraction=0.2)
    assert len(train) + len(val) == 20
    assert split(examples)[0][0].model_dump() == split(examples)[0][0].model_dump()


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = generate_synthetic(5)
    path = tmp_path / "data.jsonl"
    save_jsonl(examples, path)
    loaded = load_jsonl(path)
    assert [e.model_dump() for e in loaded] == [e.model_dump() for e in examples]


def test_bad_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_jsonl(path)


def test_example_requires_output() -> None:
    with pytest.raises(ValueError):
        InstructionExample(instruction="q")  # type: ignore[call-arg]
