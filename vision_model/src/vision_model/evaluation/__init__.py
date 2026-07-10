"""Precision/Recall/F1/mAP computation and dataset-level evaluation.

Public entry points:

    from vision_model.evaluation import evaluate, evaluate_dataset
"""

from vision_model.evaluation.evaluator import evaluate_dataset
from vision_model.evaluation.metrics import ClassMetrics, EvaluationReport, evaluate

__all__ = ["ClassMetrics", "EvaluationReport", "evaluate", "evaluate_dataset"]
