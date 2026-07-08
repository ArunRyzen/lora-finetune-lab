"""Opt-in debug logging: see the exact prompt/reply traffic behind `lora eval`.

Reading code tells you what *should* happen; watching real prompts and replies scroll by tells
you what *actually* happens. Set `LLM_DEBUG=1` — as a real environment variable or as a line in
your project's `.env` file — and every model contender in the eval (the offline fakes and, if
enabled, the live Gemini call) prints an "AI REQUEST" block with the prompt it was sent and an
"AI RESPONSE" block with what it answered.

Design notes (kept deliberately boring):
- Off by default. `LLM_DEBUG` unset, "0", or "false" (any casing) means silence.
- Two ways to switch it on, with a clear pecking order: a real env var always wins; the `.env`
  file in the current directory is only consulted when the env var is not set at all.
- The decision is cached (`functools.lru_cache`), so the `.env` file is read at most once per
  process — not once per logged prompt.
- Everything goes to **stderr**, so it never mixes with the accuracy table on stdout —
  you can still pipe or redirect the real output cleanly.
- Long fields are truncated (~2000 chars) so a huge prompt can't flood your terminal.
- Never pass secrets (API keys) as fields. Callers only log prompts and replies.
"""

from __future__ import annotations

import functools
import os
import sys

# Cap per-field output so one giant prompt doesn't bury everything else on screen.
_MAX_FIELD_CHARS = 2000


def _truthy(value: str) -> bool:
    """One shared definition of "on": anything except ""/"0"/"false" (case-insensitive).

    So `LLM_DEBUG=1`, `LLM_DEBUG=yes`, even `LLM_DEBUG=banana` all enable it;
    empty/`0`/`false` keep it off.
    """
    return value.strip().lower() not in ("", "0", "false")


def _dotenv_value() -> str | None:
    """Read LLM_DEBUG from a `.env` file in the current working directory, if possible.

    `dotenv_values` parses the file WITHOUT touching `os.environ`, which is exactly what we
    want: the real environment stays the single source of truth for precedence. The import is
    lazy and guarded — `python-dotenv` normally rides along with `pydantic-settings`, but if it
    is ever missing we simply behave as if the `.env` file did not exist.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv ships with pydantic-settings
        return None
    # "utf-8-sig" tolerates the byte-order mark some Windows editors (and PowerShell 5.1's
    # `Set-Content -Encoding utf8`) prepend; without it the BOM would glue itself onto the
    # first key and "LLM_DEBUG" would silently not be found. Identical to plain UTF-8 otherwise.
    return dotenv_values(".env", encoding="utf-8-sig").get("LLM_DEBUG")


@functools.lru_cache(maxsize=1)
def debug_enabled() -> bool:
    """Return True when LLM_DEBUG is switched on, checking two places in order.

    Precedence (first match decides — think "closest to the shell wins"):
    1. The real environment variable `LLM_DEBUG`. If it is SET (even to "0"), it alone
       decides — an explicit `LLM_DEBUG=0` in your shell silences a `.env` that says "1".
    2. Otherwise, the `LLM_DEBUG` line in a `.env` file in the current working directory.
    3. Neither present → debugging is off.

    The result is cached for the life of the process (`functools.lru_cache`), so the `.env`
    file is read at most once. Tests that flip LLM_DEBUG call `debug_enabled.cache_clear()`.
    """
    value = os.environ.get("LLM_DEBUG")
    if value is None:
        # Env var not set at all — fall back to the project's .env file (may also be absent).
        value = _dotenv_value()
    if value is None:
        return False
    return _truthy(value)


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
