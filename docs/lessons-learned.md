# Lessons Learned

Notes to my future self from building this (Milestone 5).

## Technical
- **The decision beats the trainer.** The most valuable thing here is `when-to-finetune.md`, not the
  QLoRA code. Most "we need to fine-tune" instincts are actually RAG or prompt problems — naming that
  saves weeks.
- **Split CPU core from GPU edge.** Putting the heavy stack behind a `train` extra with imports
  *inside* `run_training` meant CI, mypy, and local dev stay light and green, while the real training
  still runs on Colab. The logic (data, prompts, eval) is all tested without a GPU.
- **One formatting function, used twice.** Train/inference format drift is the classic silent quality
  killer; a single tested `format_text` removes it.
- **Eval behind a `Model` protocol pays off again.** Fake models make the base-vs-tuned-vs-RAG
  comparison testable and turn the lesson into a runnable demo.
- **Synthetic data is a real technique — with a caveat.** It bootstraps a task cheaply but inherits
  the generator's blind spots; document that, and verify.

## Process
- **No-GPU is fine for the engineering.** Everything except the train step is CPU work; the notebook
  carries the GPU part. The portfolio value is in the judgment + tested tooling, not raw compute.
- **Config-as-code.** A typed `TrainingConfig` made runs reproducible and the CLI able to print them.

## If I did it again
- Add a real-data benchmark, not just synthetic tasks.
- Wire the held-out eval into `llm-eval-kit`'s gate so a bad fine-tune fails CI.
- Add DPO after SFT and a quantized-serving (vLLM) note.
