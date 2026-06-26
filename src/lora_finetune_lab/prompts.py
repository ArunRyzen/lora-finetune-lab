"""Instruction formatting.

Fine-tuning is only as good as how consistently you format examples. We format every example the
*same* way for training and inference — a mismatch there is the most common cause of a tuned model
underperforming. This uses a simple, explicit chat template; on Colab you'd prefer the tokenizer's
own `apply_chat_template` so it matches the base model exactly (the notebook does this).
"""

from __future__ import annotations

from lora_finetune_lab.dataset import InstructionExample

SYSTEM_PROMPT = "You are a helpful, concise assistant."


def _user_turn(example: InstructionExample) -> str:
    if example.input.strip():
        return f"{example.instruction}\n\n{example.input}"
    return example.instruction


def to_messages(
    example: InstructionExample, *, include_output: bool = True
) -> list[dict[str, str]]:
    """Render an example as chat messages (the shape SFT and inference both expect)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_turn(example)},
    ]
    if include_output:
        messages.append({"role": "assistant", "content": example.output})
    return messages


def format_text(example: InstructionExample, *, include_output: bool = True) -> str:
    """A simple textual chat template — deterministic and testable.

    Training packs the full prompt + completion; inference omits the assistant turn so the model
    generates it. Same template both times.
    """
    parts = [
        f"<|system|>\n{SYSTEM_PROMPT}",
        f"<|user|>\n{_user_turn(example)}",
        "<|assistant|>\n" + (example.output if include_output else ""),
    ]
    return "\n".join(parts)
