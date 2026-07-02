# 🆓 Free GPU Guide — Colab vs Kaggle

You don't own a GPU. That's fine — for learning-scale fine-tuning you never need to
buy one. Two websites give you a real training GPU **for free, in the browser,
no credit card**. This page tells you which to pick and how to use each.

## The 30-second decision

| | **Google Colab** | **Kaggle Notebooks** |
|---|---|---|
| GPU you get | 1× NVIDIA T4 (~15 GB memory) | T4 (choose "GPU T4 x2") or P100 |
| Free allowance | A few hours/day (unpublished, varies) | **30 GPU-hours per week** (clear quota) |
| Max session length | ~ a few hours, disconnects when idle | ~9–12 hours |
| Account needs | Google account | Kaggle account + **phone verification** (required for GPU) |
| Best for | **Quick sessions** — our notebooks (10–30 min) | **Longer runs** — bigger datasets, more steps |

**Rule of thumb:** start on **Colab** (zero setup friction). The day you hit Colab's
daily limit or want a multi-hour run, move to **Kaggle** — same notebook works there.

---

## Using Google Colab (first time)

1. Go to https://colab.research.google.com and open the notebook
   (File → Open notebook → GitHub tab → paste this repo's URL, or upload the `.ipynb`).
2. **Turn on the GPU** — Runtime → *Change runtime type* → Hardware accelerator: **T4 GPU** → Save.
3. Run cells top to bottom (▶ button or `Shift+Enter`). The first install cell takes a few minutes.
4. **Save your results before closing!** The machine is wiped when the session ends:
   - Quick way: Files sidebar (📁 icon) → right-click `outputs/unsloth_adapter` → Download.
   - Better for big files: mount Google Drive —
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     # then copy: !cp -r outputs/unsloth_adapter /content/drive/MyDrive/
     ```

**Colab gotchas:** close the tab too long → disconnected; free tier can say "no GPUs
available right now" at busy times (try later); usage limits reset roughly daily.

## Using Kaggle (first time)

1. Create an account at https://www.kaggle.com and **verify your phone number**
   (Settings → Phone verification) — GPUs are locked until you do.
2. Create a notebook: *Code → New Notebook*, then File → Import Notebook → upload the `.ipynb`.
3. **Turn on the GPU** — right sidebar: *Session options → Accelerator → GPU T4 x2*
   (or P100 — either works; our notebooks use one GPU and ignore the second T4).
4. Also in the sidebar, set *Internet → On* (needed for the pip installs).
5. Run cells top to bottom. Your remaining weekly quota is shown in the sidebar.
6. **Save results:** anything written to the working directory appears under the
   notebook's **Output** tab after you click *Save Version* (Quick Save is enough).

**Kaggle gotchas:** the quota (30 h/week, resets weekly) counts *while the session is
on*, even if idle — stop the session (Power button) when you're done reading.

---

## Which GPU numbers actually matter?

- **VRAM (GPU memory, ~15–16 GB on a T4)** decides *how big a model fits*. Our trick
  stack — 4-bit loading (QLoRA) + tiny LoRA adapters + Unsloth's optimizations — is
  precisely what makes a 3B model train comfortably inside that.
- **Hours** decide *how long you can train*. Our notebooks need well under an hour.

## Others you may hear about (fine, but not needed)

- **Lightning AI** — VS-Code-like cloud IDE with free monthly GPU credits. Nice, but
  more setup than Colab/Kaggle.
- **Paid tiers** (Colab Pro, etc.) — pointless for this course; free T4s cover everything here.

> **TL;DR:** Colab for a quick session today, Kaggle for 30 guaranteed hours a week.
> Both run our notebooks unchanged: `qlora_finetune.ipynb` (classic, 0.5B) and
> `unsloth_finetune.ipynb` (Unsloth, 3B). Always download your adapter before closing.
