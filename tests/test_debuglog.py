"""LLM_DEBUG logging: silent by default, prompt/reply blocks on stderr when enabled."""

from __future__ import annotations

import pytest

from lora_finetune_lab.dataset import generate_synthetic
from lora_finetune_lab.debuglog import debug_enabled, log_block
from lora_finetune_lab.evaluation import ConstantModel, RuleModel, accuracy


def test_debug_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "False", ""])
def test_debug_disabled_for_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_debug_enabled_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is True


def test_log_block_silent_when_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    log_block("AI REQUEST (offline rule model)", prompt="What is 1 + 1?")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_log_block_writes_block_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (offline rule model)", prompt="What is 1 + 1?")
    captured = capsys.readouterr()
    assert captured.out == ""  # never pollutes stdout (the accuracy table lives there)
    assert "=== AI REQUEST (offline rule model) ===" in captured.err
    assert "prompt: What is 1 + 1?" in captured.err


def test_log_block_truncates_long_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (offline rule model)", prompt="x" * 5000)
    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 5000 not in err


def test_eval_path_silent_when_debug_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    accuracy(RuleModel(), generate_synthetic(4))
    captured = capsys.readouterr()
    assert captured.err == ""


def test_eval_path_logs_request_and_response_when_debug_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    examples = generate_synthetic(4)
    accuracy(RuleModel(), examples)
    accuracy(ConstantModel("I'm not sure."), examples)
    err = capsys.readouterr().err
    # One request/response pair per example, per model — labelled per contender.
    assert err.count("=== AI REQUEST (offline rule model) ===") == len(examples)
    assert err.count("=== AI RESPONSE (offline rule model) ===") == len(examples)
    assert err.count("=== AI REQUEST (offline constant model) ===") == len(examples)
    assert "reply: I'm not sure." in err
    # The prompt each model was actually sent (chat-formatted) is visible.
    assert "<|user|>" in err


def test_gemini_generate_logs_prompt_and_reply_but_never_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reuse the fake-SDK trick from test_gemini.py: no network, no real key.
    from tests.test_gemini import _install_fake_sdk

    _install_fake_sdk(monkeypatch, reply="42")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    monkeypatch.setenv("LLM_DEBUG", "1")

    from lora_finetune_lab.gemini import DEFAULT_GEMINI_MODEL, GeminiModel

    GeminiModel().generate("What is 40 + 2?")
    err = capsys.readouterr().err
    assert f"=== AI REQUEST (gemini/{DEFAULT_GEMINI_MODEL}) ===" in err
    assert "prompt: What is 40 + 2?" in err
    assert f"=== AI RESPONSE (gemini/{DEFAULT_GEMINI_MODEL}) ===" in err
    assert "reply: 42" in err
    assert "super-secret-key" not in err  # API keys must never be logged
