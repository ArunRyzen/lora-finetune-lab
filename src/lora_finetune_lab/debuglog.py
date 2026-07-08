"""Opt-in debug logging: see the exact prompt/reply traffic behind `lora eval`.

Reading code tells you what *should* happen; watching real prompts and replies scroll by tells
you what *actually* happens. Set the env var `LLM_DEBUG=1` and every model contender in the eval
(the offline fakes and, if enabled, the live Gemini call) prints an "AI REQUEST" block with the
prompt it was sent and an "AI RESPONSE" block with what it answered.

Design notes (kept deliberately boring):
- Off by default. `LLM_DEBUG` unset, "0", or "false" (any casing) means silence.
- Everything goes to **stderr**, so it never mixes with the accuracy table on stdout —
  you can still pipe or redirect the real output cleanly.
- Long fields are truncated (~2000 chars) so a huge prompt can't flood your terminal.
- Never pass secrets (API keys) as fields. Callers only log prompts and replies.
"""

from __future__ import annotations

import os
import sys

# Cap per-field output so one giant prompt doesn't bury everything else on screen.
_MAX_FIELD_CHARS = 2000


def debug_enabled() -> bool:
    """Return True when the LLM_DEBUG env var is set to anything truthy.

    "Truthy" here means: set and not "0"/"false" (case-insensitive). So `LLM_DEBUG=1`,
    `LLM_DEBUG=yes`, even `LLM_DEBUG=banana` all enable it; unset/`0`/`false` keep it off.
    """
    value = os.environ.get("LLM_DEBUG")
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false")


def log_block(title: str, **fields: object) -> None:
    """Print one plain-ASCII debug block to stderr (no-op unless LLM_DEBUG is on).

    Example output:

        === AI REQUEST (offline rule model) ===
        prompt: <|system|>...
    """
    if not debug_enabled():
        return
    lines = [f"=== {title} ==="]
    for name, value in fields.items():
        text = str(value)
        if len(text) > _MAX_FIELD_CHARS:
            text = text[:_MAX_FIELD_CHARS] + "... [truncated]"
        lines.append(f"{name}: {text}")
    print("\n".join(lines), file=sys.stderr, flush=True)
