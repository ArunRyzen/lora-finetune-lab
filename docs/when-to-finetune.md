# When to Fine-Tune (vs RAG vs Prompt)

The most valuable skill in this repo isn't running a trainer — it's knowing **whether you should**.
Fine-tuning is often the *wrong* first move. Here's the decision framework.

## The one-line rule
- **Prompt engineering** changes *instructions*.
- **RAG** changes *knowledge* (what facts the model can see).
- **Fine-tuning** changes *behavior* (skills, format, style, tone) — baked into the weights.

If your problem is "the model doesn't **know** X," that's RAG (or a tool), **not** fine-tuning.
If it's "the model doesn't **behave** the way I need," that's fine-tuning's territory.

## Decision order (cheapest first)
1. **Prompt / few-shot** — fastest, free, reversible. Try this first, always.
2. **RAG / tools** — when the gap is missing or fresh knowledge. No training; data stays current.
3. **Fine-tune** — only when prompting + RAG can't get the *behavior* right, and you have data.

Climb this ladder; don't jump to the top rung.

## Fine-tune when…
- You need a **consistent format/style/tone** that prompting can't reliably enforce.
- You're teaching a **skill or domain pattern** (classification style, a DSL, a structured output
  shape) that recurs across inputs.
- You want to **shrink the prompt** (move repeated instructions/examples into the weights) to cut
  per-call cost and latency at scale.
- You need a **smaller/cheaper model** to match a bigger one on a *narrow* task.

## Do NOT fine-tune when…
- The need is **knowledge/freshness** → use RAG; weights go stale, retrieval doesn't.
- You have **little data** (a few dozen examples) → prompt/few-shot will do better.
- The task **changes often** → retraining each time is slower and costlier than editing a prompt.
- You haven't **measured** that prompting/RAG fail → you might be fine-tuning a solved problem.

## Why QLoRA specifically
Full fine-tuning updates billions of weights — expensive, GPU-hungry, and produces a full model
copy. **LoRA** trains tiny low-rank adapter matrices on top of frozen weights (a few MB to ship).
**QLoRA** additionally loads the frozen base in **4-bit**, slashing memory so a 0.5–7B model trains
on a *free* Colab T4. You get ~full-fine-tune quality on a narrow task at a fraction of the cost,
and the adapter is a small artifact you can swap per task.

## How to know it worked
**Always evaluate on a held-out set**, base vs tuned vs the cheaper alternative (this repo's
`evaluation.compare`). A fine-tune that doesn't beat the base — or that a prompt tweak would have
matched — is a regression in disguise. Ship the cheapest thing that passes the bar.

## Cost & data reality (no local GPU)
- **Data is the work.** A few hundred clean, consistently-formatted examples beats thousands of
  noisy ones. **Synthetic data** (this repo's `generate_synthetic`, or a strong model drafting +
  filtering examples) bootstraps a task — but inherits the generator's blind spots, so verify it.
- **Compute** is a free Colab/Kaggle T4 for small models; rent an A100 by the hour for bigger ones.
- **Serving** a LoRA adapter is cheap: load the base once, attach the small adapter.

## TL;DR
Try prompt → RAG → fine-tune, in that order. Fine-tune to change *behavior*, not to add *knowledge*.
Use QLoRA to do it cheaply. Prove it on a held-out set before you ship.
