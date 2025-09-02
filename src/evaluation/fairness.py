"""
Fairness analysis for AI Medication Reminder evaluation.

Implements fairness metrics and bias detection according to pipeline specification A.3.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .metrics import MetricsCalculator


class FairnessAnalyzer:
    """Analyzes model fairness across different demographic and behavioral groups."""

    def __init__(self, fairness_groups: Dict[str, List[str]]):
        self.fairness_groups = fairness_groups
        self.metrics_calc = MetricsCalculator()

    def calculate_group_metrics(
        self, df: pd.DataFrame, y_true_col: str, y_proba_col: str, group_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each group within a demographic category.

        Args:
            df: DataFrame with predictions and group information
            y_true_col: Column name for true labels
            y_proba_col: Column name for predicted probabilities
            group_col: Column name for group membership

        Returns:
            Dictionary mapping group values to their metrics
        """
        group_metrics = {}

        for group_value in df[group_col].unique():
            group_mask = df[group_col] == group_value
            group_df = df[group_mask]

            if len(group_df) > 0:
                y_true = group_df[y_true_col].values
                y_proba = group_df[y_proba_col].values

                metrics = self.metrics_calc.calculate_all_metrics(y_true, y_proba)
                metrics["sample_size"] = len(group_df)
                metrics["positive_rate"] = y_true.mean()

                group_metrics[str(group_value)] = metrics

        return group_metrics

    def calculate_fairness_metrics(
        self, group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate fairness metrics across groups.

        Args:
            group_metrics: Dictionary of metrics for each group

        Returns:
            Dictionary of fairness metrics
        """
        if len(group_metrics) < 2:
            return {"error": "Need at least 2 groups for fairness analysis"}  # type: ignore

        # Extract key metrics for each group
        group_names = list(group_metrics.keys())
        accuracies = [group_metrics[g]["accuracy"] for g in group_names]
        precisions = [group_metrics[g]["precision"] for g in group_names]
        recalls = [group_metrics[g]["recall"] for g in group_names]
        f1_scores = [group_metrics[g]["f1"] for g in group_names]
        positive_rates = [group_metrics[g]["positive_rate"] for g in group_names]

        # Calculate fairness metrics
        fairness_metrics = {
            # Equalized odds (difference in TPR and FPR)
            "max_accuracy_diff": max(accuracies) - min(accuracies),
            "max_precision_diff": max(precisions) - min(precisions),
            "max_recall_diff": max(recalls) - min(recalls),
            "max_f1_diff": max(f1_scores) - min(f1_scores),
            # Demographic parity (difference in positive prediction rates)
            "max_positive_rate_diff": max(positive_rates) - min(positive_rates),
            # Statistical measures
            "accuracy_std": np.std(accuracies),
            "precision_std": np.std(precisions),
            "recall_std": np.std(recalls),
            "f1_std": np.std(f1_scores),
            # Overall fairness score (lower is better)
            "fairness_score": np.mean(
                [
                    max(accuracies) - min(accuracies),
                    max(precisions) - min(precisions),
                    max(recalls) - min(recalls),
                    max(f1_scores) - min(f1_scores),
                ]
            ),
        }

        return fairness_metrics

    def analyze_all_groups(
        self, df: pd.DataFrame, y_true_col: str, y_proba_col: str
    ) -> Dict[str, Dict]:
        """
        Analyze fairness across all configured demographic groups.

        Args:
            df: DataFrame with predictions and group information
            y_true_col: Column name for true labels
            y_proba_col: Column name for predicted probabilities

        Returns:
            Dictionary with fairness analysis for each demographic category
        """
        fairness_analysis = {}

        for group_category, group_values in self.fairness_groups.items():
            if group_category in df.columns:
                print(f"Analyzing fairness for {group_category}...")

                # Calculate metrics for each group
                group_metrics = self.calculate_group_metrics(
                    df, y_true_col, y_proba_col, group_category
                )

                # Calculate fairness metrics
                fairness_metrics = self.calculate_fairness_metrics(group_metrics)

                fairness_analysis[group_category] = {
                    "group_metrics": group_metrics,
                    "fairness_metrics": fairness_metrics,
                }
            else:
                print(f"Warning: {group_category} column not found in data")

        return fairness_analysis

    def print_fairness_summary(self, fairness_analysis: Dict[str, Dict]) -> None:
        """
        Print a formatted summary of fairness analysis.

        Args:
            fairness_analysis: Results from analyze_all_groups
        """
        print("=" * 70)
        print("FAIRNESS ANALYSIS SUMMARY")
        print("=" * 70)

        for category, analysis in fairness_analysis.items():
            print(f"\n{category.upper()} FAIRNESS:")
            print("-" * 40)

            group_metrics = analysis["group_metrics"]
            fairness_metrics = analysis["fairness_metrics"]

            # Print group-level metrics
            print("Group Performance:")
            for group, metrics in group_metrics.items():
                print(
                    f"  {group}: Acc={metrics['accuracy']:.3f}, "
                    f"Prec={metrics['precision']:.3f}, "
                    f"Rec={metrics['recall']:.3f}, "
                    f"F1={metrics['f1']:.3f}, "
                    f"N={metrics['sample_size']}"
                )

            # Print fairness metrics
            print("\nFairness Metrics:")
            print(
                f"  Max Accuracy Diff:  {fairness_metrics.get('max_accuracy_diff', 0):.4f}"
            )
            print(
                f"  Max Precision Diff: {fairness_metrics.get('max_precision_diff', 0):.4f}"
            )
            print(
                f"  Max Recall Diff:    {fairness_metrics.get('max_recall_diff', 0):.4f}"
            )
            print(f"  Max F1 Diff:        {fairness_metrics.get('max_f1_diff', 0):.4f}")
            print(
                f"  Overall Fairness:   {fairness_metrics.get('fairness_score', 0):.4f}"
            )

            # Fairness assessment
            fairness_score = fairness_metrics.get("fairness_score", 0)
            if fairness_score < 0.05:
                assessment = "GOOD"
            elif fairness_score < 0.10:
                assessment = "MODERATE"
            else:
                assessment = "POOR"
            print(f"  Assessment:         {assessment}")

        print("=" * 70)
