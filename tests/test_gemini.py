"""Gemini adapter: fully offline tests — the SDK is faked, no network, no real key."""

from __future__ import annotations

import sys
import types

import pytest

from lora_finetune_lab.errors import GeminiDependencyError
from lora_finetune_lab.gemini import DEFAULT_GEMINI_MODEL, GeminiModel


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, reply: str | None) -> dict[str, object]:
    """Insert a fake `google.genai` into sys.modules so GeminiModel's lazy import finds it."""
    calls: dict[str, object] = {}

    class FakeResponse:
        text = reply

    class FakeModels:
        def generate_content(self, *, model: str, contents: str) -> FakeResponse:
            calls["model"] = model
            calls["contents"] = contents
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.models = FakeModels()

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = FakeClient  # type: ignore[attr-defined]
    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    return calls


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiDependencyError, match="GEMINI_API_KEY"):
        GeminiModel()


def test_generate_calls_sdk_and_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_sdk(monkeypatch, reply="42")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = GeminiModel()
    assert model.generate("What is 40 + 2?") == "42"
    assert calls["api_key"] == "test-key"
    assert calls["model"] == DEFAULT_GEMINI_MODEL
    assert calls["contents"] == "What is 40 + 2?"


def test_generate_returns_empty_string_for_blocked_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch, reply=None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert GeminiModel().generate("anything") == ""
