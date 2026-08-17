"""Evaluation helpers for financial retrieval metrics."""

from importlib import import_module

__all__ = [
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "analyze_history_false_positives",
    "analyze_retrieval_errors",
    "evaluate",
    "evaluate_batch",
    "load_jsonl_results",
    "load_history_results",
    "print_overall_evaluation",
    "evaluate_history_db",
]


def __getattr__(name: str):
    if name in __all__:
        metrics = import_module(".metrics", __name__)
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
