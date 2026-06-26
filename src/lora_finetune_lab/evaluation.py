"""Evaluate base vs fine-tuned vs RAG — the comparison that justifies (or kills) a fine-tune.

You never ship a fine-tune on faith: you measure it against the un-tuned base *and* against cheaper
alternatives (RAG, prompting) on a held-out set. This harness does that with a `Model` protocol, so
the same code scores a real model on Colab or a fake model in CI. The demo models below illustrate
the key lesson: fine-tuning teaches *skills/format*; RAG supplies *knowledge* — they fix different
problems.
"""

from __future__ import annotations

import re
from typing import Protocol

from lora_finetune_lab.dataset import InstructionExample
from lora_finetune_lab.prompts import format_text


class Model(Protocol):
    def generate(self, prompt: str) -> str: ...


def _normalize(text: str) -> str:
    return text.strip().lower()


def accuracy(model: Model, examples: list[InstructionExample]) -> float:
    """Fraction of examples whose expected output appears in the model's generation."""
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        prompt = format_text(ex, include_output=False)
        if _normalize(ex.output) in _normalize(model.generate(prompt)):
            correct += 1
    return correct / len(examples)


def compare(models: dict[str, Model], examples: list[InstructionExample]) -> dict[str, float]:
    """Score each named model on the same held-out set."""
    return {name: accuracy(model, examples) for name, model in models.items()}


# --- Demo / test models --------------------------------------------------------------


class ConstantModel:
    """Returns a fixed string regardless of input — an untuned base / unhelpful RAG stand-in."""

    def __init__(self, reply: str = "I'm not sure.") -> None:
        self._reply = reply

    def generate(self, prompt: str) -> str:
        return self._reply


class RuleModel:
    """Actually solves the synthetic tasks — stands in for a model that *learned* the skill."""

    def generate(self, prompt: str) -> str:
        user = self._user_turn(prompt)
        add = re.search(r"what is (\d+)\s*\+\s*(\d+)", user.lower())
        if add:
            return str(int(add.group(1)) + int(add.group(2)))
        body = user.split("\n\n", 1)
        payload = body[1].strip() if len(body) > 1 else ""
        if "uppercase" in user.lower():
            return payload.upper()
        if "reverse" in user.lower():
            return payload[::-1]
        return "I'm not sure."

    @staticmethod
    def _user_turn(prompt: str) -> str:
        match = re.search(r"<\|user\|>\n(.*?)\n<\|assistant\|>", prompt, re.S)
        return match.group(1) if match else prompt
