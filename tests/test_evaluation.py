"""Evaluation: the 'fine-tuned' model beats the untuned/RAG baselines on the synthetic skill."""

from __future__ import annotations

from lora_finetune_lab.dataset import generate_synthetic
from lora_finetune_lab.evaluation import ConstantModel, RuleModel, accuracy, compare


def test_rule_model_solves_the_task() -> None:
    examples = generate_synthetic(30)
    assert accuracy(RuleModel(), examples) >= 0.9


def test_constant_model_fails_the_task() -> None:
    examples = generate_synthetic(30)
    assert accuracy(ConstantModel("nope"), examples) == 0.0


def test_compare_ranks_finetuned_above_baselines() -> None:
    examples = generate_synthetic(40)
    scores = compare(
        {"base": ConstantModel(), "tuned": RuleModel(), "rag": ConstantModel("docs")}, examples
    )
    assert scores["tuned"] > scores["base"]
    assert scores["tuned"] > scores["rag"]
