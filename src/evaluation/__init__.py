"""
Evaluation module for AI Medication Reminder.

This module handles model evaluation, metrics calculation, and fairness analysis.
"""

from .evaluator import ModelEvaluator
from .fairness import FairnessAnalyzer
from .metrics import MetricsCalculator

__all__ = ["ModelEvaluator", "FairnessAnalyzer", "MetricsCalculator"]
