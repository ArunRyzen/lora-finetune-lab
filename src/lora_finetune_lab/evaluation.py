"""Evaluate base vs fine-tuned vs RAG — the comparison that justifies (or kills) a fine-tune.

You never ship a fine-tune on faith: you measure it against the un-tuned base *and* against
cheaper alternatives (RAG, prompting) on a held-out set. This harness does that, and the key
design trick is the `Model` protocol below: anything with a `generate(prompt) -> str` method can
enter the comparison — a real GPU model on Colab, a hosted API model, or a fake model in CI.
The scoring code never knows the difference.

The demo models at the bottom illustrate the repo's central lesson: fine-tuning teaches
*skills/format* (behaviour); RAG supplies *knowledge* (facts) — they fix different problems.
See `docs/when-to-finetune.md`.
"""

from __future__ import annotations

import re
from typing import Protocol

from lora_finetune_lab.dataset import InstructionExample
from lora_finetune_lab.prompts import format_text


class Model(Protocol):
    """The one thing a model must do to be evaluated: take a prompt, return a string.

    A Protocol is "duck typing with a contract": any class with a matching `generate` method
    counts as a Model automatically — no inheritance or registration needed. That's why real
    Hugging Face models (notebook), the Gemini API adapter, and the fakes below all plug into
    the same `accuracy`/`compare` functions.
    """

    def generate(self, prompt: str) -> str: ...


def _normalize(text: str) -> str:
    # Trim whitespace and lowercase before comparing, so "  Amall " still matches "amall".
    return text.strip().lower()


def accuracy(model: Model, examples: list[InstructionExample]) -> float:
    """Fraction of examples whose expected output appears in the model's generation."""
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        # include_output=False: give the model the question WITHOUT the answer — it must
        # generate the answer itself (exactly like a student sitting the exam).
        prompt = format_text(ex, include_output=False)
        # Lenient grading: the expected answer just has to appear somewhere in the reply,
        # so "The answer is 42." still counts when we expected "42".
        if _normalize(ex.output) in _normalize(model.generate(prompt)):
            correct += 1
    return correct / len(examples)


def compare(models: dict[str, Model], examples: list[InstructionExample]) -> dict[str, float]:
    """Score each named model on the same held-out set.

    Same questions for every contender — that's what makes the numbers comparable. This is the
    function `lora eval` calls to produce the base vs fine-tuned vs RAG table.
    """
    return {name: accuracy(model, examples) for name, model in models.items()}


# --- Demo / test models --------------------------------------------------------------
# These fakes let the whole eval pipeline run (and be tested) with no GPU and no network.
# Each one plays a role in the base-vs-tuned-vs-RAG story told by `lora eval`.


class ConstantModel:
    """Returns a fixed string regardless of input — an untuned base / unhelpful RAG stand-in.

    Why does this stand in for RAG? RAG = "look up relevant documents, paste them into the
    prompt". Our synthetic tasks (add two numbers, reverse a string) need a *skill*, not a
    fact, so there is nothing useful to retrieve — retrieval comes back empty-handed every
    time, which a constant "not found" reply models perfectly.
    """

    def __init__(self, reply: str = "I'm not sure.") -> None:
        self._reply = reply

    def generate(self, prompt: str) -> str:
        return self._reply


class RuleModel:
    """Actually solves the synthetic tasks — stands in for a model that *learned* the skill.

    It's just if/else rules, but from the eval harness's point of view it behaves exactly like
    a successfully fine-tuned model would: reads the task, produces the right answer.
    """

    def generate(self, prompt: str) -> str:
        # First, pull the user's request out of the chat-formatted prompt (see _user_turn).
        user = self._user_turn(prompt)
        # Task 1: "What is 12 + 34?" — capture the two numbers and add them.
        add = re.search(r"what is (\d+)\s*\+\s*(\d+)", user.lower())
        if add:
            return str(int(add.group(1)) + int(add.group(2)))
        # Tasks 2 & 3 carry their payload after a blank line ("Reverse...\n\nllama").
        body = user.split("\n\n", 1)
        payload = body[1].strip() if len(body) > 1 else ""
        if "uppercase" in user.lower():
            return payload.upper()
        if "reverse" in user.lower():
            return payload[::-1]
        return "I'm not sure."

    @staticmethod
    def _user_turn(prompt: str) -> str:
        # The eval sends the full chat template ("<|system|>...<|user|>...<|assistant|>");
        # grab just the text between the user and assistant markers.
        match = re.search(r"<\|user\|>\n(.*?)\n<\|assistant\|>", prompt, re.S)
        return match.group(1) if match else prompt
