"""Instruction formatting: turn a dataset example into the exact text the model sees.

Chat models don't see "an instruction" and "an answer" as separate things — they see one long
string with special markers (a *chat template*) showing who said what. This module produces that
string.

The golden rule: format examples **the same way for training and for inference**. If the model
was trained seeing `<|user|>\\n...` but you later prompt it without those markers, it's like
studying from one textbook and being examined from another — the most common reason a tuned
model mysteriously underperforms.

We use a simple, explicit template here so it's deterministic and easy to test. On Colab you'd
prefer the tokenizer's own `apply_chat_template`, which matches the base model's native format
exactly (the notebook does this).
"""

from __future__ import annotations

from lora_finetune_lab.dataset import InstructionExample

# The standing instruction prepended to every conversation. Keep it identical everywhere.
SYSTEM_PROMPT = "You are a helpful, concise assistant."


def _user_turn(example: InstructionExample) -> str:
    # Merge the instruction and the optional input into one user message, e.g.
    # "Reverse the input string.\n\nllama". A blank line separates task from payload.
    if example.input.strip():
        return f"{example.instruction}\n\n{example.input}"
    return example.instruction


def to_messages(
    example: InstructionExample, *, include_output: bool = True
) -> list[dict[str, str]]:
    """Render an example as chat messages (the shape SFT and inference both expect)."""
    # This list-of-dicts shape ({"role": ..., "content": ...}) is the standard "conversation"
    # format that both training libraries and chat APIs understand.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_turn(example)},
    ]
    if include_output:
        # The assistant's turn is the ANSWER — included for training, omitted when asking.
        messages.append({"role": "assistant", "content": example.output})
    return messages


def format_text(example: InstructionExample, *, include_output: bool = True) -> str:
    """A simple textual chat template — deterministic and testable.

    Training packs the full prompt + completion; inference omits the assistant turn so the model
    generates it. Same template both times.
    """
    # include_output=True  -> "<|assistant|>\n<the answer>"   (training: model learns the answer)
    # include_output=False -> "<|assistant|>\n"               (inference: model fills it in)
    parts = [
        f"<|system|>\n{SYSTEM_PROMPT}",
        f"<|user|>\n{_user_turn(example)}",
        "<|assistant|>\n" + (example.output if include_output else ""),
    ]
    return "\n".join(parts)
