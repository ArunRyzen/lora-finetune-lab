"""Shared pytest fixtures — keep the suite hermetic against a developer's real `.env`.

`debug_enabled()` now also reads `LLM_DEBUG` from a `.env` file in the current directory, and a
developer may legitimately keep `LLM_DEBUG=1` in theirs. Without a guard, that file would leak
into every test that expects debugging to be OFF. So, for every test, we pin the real env var to
"0" — which outranks any `.env` by design — and clear the `lru_cache` on `debug_enabled()` so no
cached answer bleeds from one test into the next. Tests that want debugging ON simply
`monkeypatch.setenv("LLM_DEBUG", "1")` on top of this baseline.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from lora_finetune_lab.debuglog import debug_enabled


@pytest.fixture(autouse=True)
def _hermetic_llm_debug(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_DEBUG", "0")  # real env var wins over any .env file
    debug_enabled.cache_clear()  # forget decisions made before/by other tests
    yield
    debug_enabled.cache_clear()  # don't let this test's decision leak forward
