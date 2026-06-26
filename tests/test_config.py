"""Training config: defaults, derived values, and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lora_finetune_lab.config import TrainingConfig


def test_defaults_and_effective_batch_size() -> None:
    cfg = TrainingConfig()
    assert cfg.lora_r == 16
    assert cfg.load_in_4bit is True
    assert cfg.effective_batch_size == cfg.per_device_batch_size * cfg.grad_accum_steps


def test_rejects_invalid_dropout() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(lora_dropout=1.5)


def test_serializable() -> None:
    cfg = TrainingConfig(base_model="my/model")
    assert "my/model" in cfg.model_dump_json()
