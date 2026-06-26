"""Prompt formatting must be consistent and reversible enough to parse back."""

from __future__ import annotations

from lora_finetune_lab.dataset import InstructionExample
from lora_finetune_lab.prompts import format_text, to_messages


def test_format_includes_instruction_and_output() -> None:
    ex = InstructionExample(instruction="Reverse the input.", input="abc", output="cba")
    text = format_text(ex)
    assert "Reverse the input." in text
    assert "abc" in text
    assert text.rstrip().endswith("cba")


def test_inference_format_omits_output() -> None:
    ex = InstructionExample(instruction="Say hi", output="hi")
    text = format_text(ex, include_output=False)
    assert text.rstrip().endswith("<|assistant|>")


def test_to_messages_roles() -> None:
    ex = InstructionExample(instruction="q", output="a")
    roles = [m["role"] for m in to_messages(ex)]
    assert roles == ["system", "user", "assistant"]
