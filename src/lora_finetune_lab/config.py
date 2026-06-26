"""Training configuration — typed, validated, and serializable.

Defaults target a small instruct model that fits QLoRA on a free Colab T4. Every hyperparameter is
here in one place so a run is reproducible and diffable (config-as-code).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """QLoRA training hyperparameters."""

    # A small instruct model that QLoRA-fits on a free T4. Swap for your target model.
    base_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    output_dir: str = Field(default="outputs/adapter")

    # LoRA: train tiny low-rank adapters instead of the full model. r is the bottleneck rank.
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # QLoRA: load the frozen base in 4-bit to slash memory; adapters train in higher precision.
    load_in_4bit: bool = Field(default=True)

    # Optimization.
    learning_rate: float = Field(default=2e-4, gt=0.0)
    epochs: int = Field(default=3, ge=1)
    per_device_batch_size: int = Field(default=2, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    max_seq_len: int = Field(default=1024, ge=64)
    seed: int = Field(default=42)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum_steps
