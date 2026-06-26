"""lora-finetune-lab: a small, honest QLoRA fine-tuning workflow.

The training itself runs on a GPU (Colab/Kaggle) via `notebooks/qlora_finetune.ipynb`. Everything
*around* training — instruction formatting, dataset building, synthetic data, and base-vs-tuned
evaluation — is plain CPU Python here, so it's tested in CI. The headline deliverable is judgment:
**when to fine-tune vs RAG vs prompt** (see `docs/when-to-finetune.md`).
"""

from lora_finetune_lab.config import TrainingConfig
from lora_finetune_lab.dataset import InstructionExample

__version__ = "0.1.0"

__all__ = ["TrainingConfig", "InstructionExample", "__version__"]
