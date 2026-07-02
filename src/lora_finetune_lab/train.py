"""The actual QLoRA training step (GPU).

Heavy imports live inside the function so the rest of the package (and CI) never needs torch. Run
this on Colab/Kaggle via the notebook, or `pip install '.[train]'` on a GPU box. The shape:
1. Load the frozen base model in **4-bit** (QLoRA) — this is what fits a 0.5–7B model on a free T4.
2. Attach small **LoRA adapters** to the attention projections — only these train.
3. Supervised fine-tune on the formatted texts, then save just the adapter (a few MB).
"""

from __future__ import annotations

from lora_finetune_lab.config import TrainingConfig
from lora_finetune_lab.errors import TrainingDependencyError


def run_training(config: TrainingConfig, train_texts: list[str]) -> str:
    """Fine-tune with QLoRA and save the adapter. Returns the output directory."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # CI / CPU box without the training stack
        raise TrainingDependencyError(
            "Training requires the GPU stack. Use the Colab notebook or `pip install '.[train]'`."
        ) from exc

    # Step 1 — the "Q" in QLoRA: describe HOW to compress the frozen base model to 4-bit
    # (nf4 = a 4-bit format designed for neural-net weights; double quant squeezes a bit more).
    quant = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # The tokenizer turns text into token IDs. Some models ship without a padding token
    # (used to make batch rows the same length), so we reuse the end-of-sequence token.
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Download the base model, compressed to 4-bit as configured above; "auto" puts it on GPU.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, quantization_config=quant, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Step 2 — LoRA: describe the tiny trainable adapters to bolt onto the frozen model.
    # Only these adapters get gradient updates; the billions of base weights never change.
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Step 3 — supervised fine-tuning (SFT): show the model our formatted examples and let the
    # trainer handle batching, gradient accumulation, and the optimization loop.
    dataset = Dataset.from_dict({"text": train_texts})
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        max_seq_length=config.max_seq_len,
        seed=config.seed,
        logging_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
    )
    trainer.train()
    # Save ONLY the adapter (a few MB), not a full copy of the base model.
    trainer.save_model(config.output_dir)
    return config.output_dir
