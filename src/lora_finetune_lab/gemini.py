"""Optional Gemini-backed model — one extra comparison point for `lora eval`.

Why would a *local fine-tuning* repo talk to an API model at all? Because the honest question
behind every fine-tune is: "would a strong general model, just prompted, already do this?" If
Gemini nails your task with zero training, that is a strong hint that prompting a hosted model is
the cheaper answer (see `docs/when-to-finetune.md`, "Decision order").

This is deliberately tiny and entirely optional:
- No API key, no SDK installed → the rest of the repo works exactly as before, fully offline.
- Same pattern as `train.py`: the heavy/optional import happens *inside* the class, so importing
  this module (or the package) never requires the `google-genai` SDK.
- It satisfies the same `Model` protocol as everything else in `evaluation.py` — the eval code
  doesn't know or care that this one calls a remote API.
"""

from __future__ import annotations

import os

from lora_finetune_lab.debuglog import log_block
from lora_finetune_lab.errors import GeminiDependencyError

# A fast, cheap hosted model — good enough to answer "would prompting alone solve this?".
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiModel:
    """Adapts the Gemini API to the repo's `Model` protocol (one `generate` method)."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None) -> None:
        # Check the key *before* importing the SDK, so the error message tells you the real
        # problem (no key vs. no package) without needing the package installed.
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GeminiDependencyError(
                "Set GEMINI_API_KEY to enable the Gemini comparison in `lora eval`."
            )
        try:
            # Imported lazily (like torch in train.py) so the core package stays light.
            from google import genai
        except ImportError as exc:
            raise GeminiDependencyError(
                "The google-genai SDK is not installed. Run: pip install '.[gemini]'"
            ) from exc
        self._client = genai.Client(api_key=key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Send the prompt to Gemini and return its text reply (matches the Model protocol)."""
        # LLM_DEBUG=1 shows the exact prompt/reply traffic (see debuglog.py). The API key is
        # deliberately never logged — only the prompt and the reply.
        log_block(f"AI REQUEST (gemini/{self._model})", prompt=prompt)
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        # The SDK returns None for empty/blocked responses; the eval expects a plain string.
        reply = response.text or ""
        log_block(f"AI RESPONSE (gemini/{self._model})", reply=reply)
        return reply
