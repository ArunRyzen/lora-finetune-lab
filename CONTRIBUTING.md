# Contributing

## Setup
```bash
uv sync --extra dev          # CPU dev: lint, types, tests
uv run pre-commit install
# For training on a GPU box:
uv sync --extra train
```

## Checks (CI enforces these — CPU only)
```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
```

## Conventions
- Keep the core CPU-light: training-only deps belong in the `train` extra with imports inside
  functions, so CI and `mypy` never need torch.
- Type hints everywhere; tests for data/prompt/eval logic (no GPU, no network).
- Secrets via `.env` (never committed). Conventional-commit messages.
