"""Domain exceptions."""

from __future__ import annotations


class LoraLabError(Exception):
    """Base class."""


class DatasetError(LoraLabError):
    """A dataset could not be loaded or was malformed."""


class TrainingDependencyError(LoraLabError):
    """GPU training stack not installed — use the Colab notebook or `pip install '.[train]'`."""


class GeminiDependencyError(LoraLabError):
    """Gemini comparison not available — needs `pip install '.[gemini]'` and GEMINI_API_KEY set."""
