"""LLM_DEBUG logging: silent by default, prompt/reply blocks on stderr when enabled.

Note on hermeticity: an autouse fixture in conftest.py pins the real env var to "0" for every
test. Tests below that need the env var truly UNSET (to exercise the `.env` fallback) delenv it
AND chdir into a temp directory, so a developer's real `.env` at the repo root can't interfere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lora_finetune_lab.dataset import generate_synthetic
from lora_finetune_lab.debuglog import debug_enabled, log_block
from lora_finetune_lab.evaluation import ConstantModel, RuleModel, accuracy


def test_debug_disabled_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # empty dir: no .env fallback to consult
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "False", ""])
def test_debug_disabled_for_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_debug_enabled_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled() is True


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])  # -sig = Windows editors' BOM
def test_dotenv_file_enables_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encoding: str
) -> None:
    """With the env var unset, `LLM_DEBUG=1` in ./.env switches debugging on."""
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding=encoding)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert debug_enabled() is True


def test_env_var_beats_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real env var — even an explicit "0" — outranks whatever the .env file says."""
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_DEBUG", "0")
    debug_enabled.cache_clear()
    assert debug_enabled() is False


def test_debug_disabled_when_no_env_var_and_no_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither the env var nor a .env file present → debugging stays off."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
    assert debug_enabled() is False


def test_log_block_silent_when_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    debug_enabled.cache_clear()
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
