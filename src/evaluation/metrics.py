"""
Metrics calculation for AI Medication Reminder evaluation.

Implements comprehensive metrics according to pipeline specification A.3.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MetricsCalculator:
    """Calculates comprehensive evaluation metrics for model performance."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of basic metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_proba)
            if len(np.unique(y_true)) > 1
            else 0.0,
        }

        return metrics

    def calculate_confusion_matrix_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics derived from confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of confusion matrix metrics
        """
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            metrics = {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "positive_predictive_value": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            }
        else:
            # Handle edge cases
            metrics = {
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "specificity": 0.0,
                "sensitivity": 0.0,
                "positive_predictive_value": 0.0,
                "negative_predictive_value": 0.0,
            }

        return metrics

    def calculate_calibration_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Calculate calibration metrics including Brier score and reliability.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Dictionary of calibration metrics
        """
        # Brier score
        brier_score = np.mean((y_proba - y_true) ** 2)

        # Binned calibration
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0  # Expected Calibration Error
        mce = 0  # Maximum Calibration Error

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()

                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)

        return {
            "brier_score": brier_score,  # type: ignore
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
        }

    def calculate_threshold_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics at optimal thresholds.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of threshold-based metrics
        """
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

        # Optimal threshold by Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

        # F1 scores for different thresholds
        f1_scores = (
            2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        )
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = (
            pr_thresholds[optimal_f1_idx] if len(pr_thresholds) > 0 else 0.5
        )

        return {
            "optimal_threshold_youden": optimal_threshold,
            "optimal_threshold_f1": optimal_f1_threshold,
            "max_youden_j": j_scores[optimal_idx],
            "max_f1_score": f1_scores[optimal_f1_idx] if len(f1_scores) > 0 else 0.0,
        }

    def calculate_all_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Classification threshold (uses instance default if None)

        Returns:
            Dictionary containing all metrics
        """
        if threshold is None:
            threshold = self.threshold

        # Convert probabilities to predictions
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate all metric categories
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred, y_proba)
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_proba)
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_proba)

        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            **cm_metrics,
            **calibration_metrics,
            **threshold_metrics,
            "threshold_used": threshold,
        }

        return all_metrics

    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """
        Print a formatted summary of metrics.

        Args:
            metrics: Dictionary of calculated metrics
        """
        print("=" * 60)
        print("MODEL EVALUATION METRICS")
        print("=" * 60)

        print("\nBasic Classification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
        print(f"  AUC:       {metrics.get('auc', 0):.4f}")

        print("\nConfusion Matrix Metrics:")
        print(f"  True Positives:  {metrics.get('true_positives', 0)}")
        print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
        print(f"  False Positives: {metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {metrics.get('false_negatives', 0)}")
        print(f"  Specificity:     {metrics.get('specificity', 0):.4f}")
        print(f"  Sensitivity:     {metrics.get('sensitivity', 0):.4f}")

        print("\nCalibration Metrics:")
        print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"  ECE:         {metrics.get('expected_calibration_error', 0):.4f}")
        print(f"  MCE:         {metrics.get('maximum_calibration_error', 0):.4f}")

        print("\nOptimal Thresholds:")
        print(
            f"  Youden J:    {metrics.get('optimal_threshold_youden', 0):.4f} (J={metrics.get('max_youden_j', 0):.4f})"
        )
        print(
            f"  Max F1:      {metrics.get('optimal_threshold_f1', 0):.4f} (F1={metrics.get('max_f1_score', 0):.4f})"
        )
        print(f"  Used:        {metrics.get('threshold_used', 0.5):.4f}")

        print("=" * 60)
