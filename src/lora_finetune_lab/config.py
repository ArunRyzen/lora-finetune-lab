"""Training configuration — typed, validated, and serializable.

Every knob that affects a training run lives in this ONE class. Why bother?
- **Reproducible:** print the config, save it next to the results, and you can rerun the exact
  same experiment later ("config-as-code").
- **Validated:** Pydantic checks each value (e.g. `ge=1` means "must be >= 1") the moment the
  config is created, so a typo fails loudly instead of wasting an hour of GPU time.

Defaults target a small instruct model that fits QLoRA on a free Colab T4.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """QLoRA training hyperparameters."""

    # Which model to start from. "Instruct" means it already follows instructions — we only
    # nudge its behaviour, we don't teach it English. 0.5B parameters = small enough for free GPUs.
    base_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    # Where the trained adapter (the tiny add-on, a few MB — not a full model copy) gets saved.
    output_dir: str = Field(default="outputs/adapter")

    # --- LoRA settings -------------------------------------------------------------------
    # LoRA = freeze the whole base model and train tiny "adapter" matrices bolted on top.
    # `r` (rank) is the adapter's size/capacity: bigger r = more trainable weights = more
    # expressive but slower and more prone to overfitting. 8–32 is the usual range.
    lora_r: int = Field(default=16, ge=1)
    # Scaling factor for how strongly the adapter's output is mixed into the frozen model.
    # Rule of thumb: alpha = 2 * r.
    lora_alpha: int = Field(default=32, ge=1)
    # Randomly zero out 5% of adapter activations during training — a standard overfitting guard.
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    # WHERE to bolt the adapters on: the attention projection layers (query/key/value/output).
    # These are the layers most responsible for "how the model attends to the prompt".
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # --- The "Q" in QLoRA ----------------------------------------------------------------
    # Load the frozen base model in 4-bit precision (a quarter of the usual memory). The base
    # is only *read* during training, so lower precision there is acceptable; the small adapters
    # still train in higher precision. This is the trick that makes a free T4 GPU enough.
    load_in_4bit: bool = Field(default=True)

    # --- Optimization --------------------------------------------------------------------
    # How big a step the optimizer takes each update. 2e-4 is a common LoRA default —
    # higher than full fine-tuning uses, because only the tiny adapters are moving.
    learning_rate: float = Field(default=2e-4, gt=0.0)
    # How many full passes over the training data. Few epochs: small datasets memorize fast.
    epochs: int = Field(default=3, ge=1)
    # Examples processed at once on the GPU. Kept tiny so it fits in the T4's 16 GB memory.
    per_device_batch_size: int = Field(default=2, ge=1)
    # Trick: accumulate gradients over 4 small batches before updating, which behaves like a
    # batch of 8 (2 x 4) without needing the memory for 8 examples at once.
    grad_accum_steps: int = Field(default=4, ge=1)
    # Longest example (in tokens) we train on; longer ones get truncated. Longer = more memory.
    max_seq_len: int = Field(default=1024, ge=64)
    # Fixed random seed so shuffling etc. is repeatable — rerunning gives the same result.
    seed: int = Field(default=42)

    @property
    def effective_batch_size(self) -> int:
        # The batch size the optimizer effectively "sees" (see grad_accum_steps above).
        return self.per_device_batch_size * self.grad_accum_steps
