# Fine-Tuning Interview Questions This Project Answers

---

### Q. Fine-tune vs RAG vs prompt — how do you choose?
Prompting changes *instructions*, RAG changes *knowledge*, fine-tuning changes *behavior* (skills,
format, style). Climb the ladder cheapest-first: prompt → RAG → fine-tune. If the gap is "doesn't
*know* X," that's RAG; if it's "doesn't *behave* how I need," that's fine-tuning. Don't fine-tune to
add facts — weights go stale, retrieval doesn't.

### Q. What is LoRA, and what is QLoRA?
**LoRA** freezes the base weights and trains small low-rank adapter matrices on top — you ship a few
MB instead of a full model copy, and training is far cheaper. **QLoRA** adds 4-bit quantization of
the frozen base, cutting memory enough to fine-tune a 0.5–7B model on a free Colab T4. Near-full
quality on a narrow task at a fraction of the cost.

### Q. When should you NOT fine-tune?
When the need is knowledge/freshness (use RAG), when you have little data (prompt/few-shot wins),
when the task changes often (retraining churn), or when you haven't measured that prompting/RAG fail.
Fine-tuning a solved problem is a cost with no benefit.

### Q. How do you know a fine-tune actually helped?
Evaluate on a **held-out set**: base vs tuned vs the cheaper alternative (RAG/prompt). If tuned
doesn't beat the base — or a prompt tweak would have matched it — don't ship. This repo's
`evaluation.compare` does exactly this; the demo shows base 0.0 vs tuned 1.0 on a skill RAG can't fix.

### Q. Why is formatting consistency important?
The model learns the exact prompt→completion shape you train on. If inference formats differently
(different template, system prompt, separators), quality drops. Use one formatting function for both
(here, `prompts.format_text`) — ideally the tokenizer's own chat template so it matches the base.

### Q. Where does the data come from, and how much do you need?
Data is the real work. A few hundred clean, consistently-formatted examples usually beat thousands of
noisy ones. **Synthetic data** bootstraps a task — a strong model drafts examples, then you filter
and verify — but it inherits the generator's blind spots, so a verification pass matters.

### Q. How do you serve a fine-tuned model cheaply?
Serve the base once and attach the small **LoRA adapter** (swap adapters per task). For throughput,
merge the adapter and serve quantized (e.g., vLLM). The adapter being tiny is a deployment advantage,
not just a training one.

### Q. What hyperparameters matter for LoRA?
Rank `r` (capacity of the adapter), `lora_alpha` (scaling), `target_modules` (usually the attention
projections), learning rate, and epochs. Start small (`r=16`, `alpha=32`), watch for overfitting on a
small dataset, and let the held-out eval — not the training loss — decide.
