"""
Model evaluation orchestrator for AI Medication Reminder.

Handles comprehensive model evaluation including metrics, fairness, and stress testing.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Suppress specific numpy warnings that don't affect functionality
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)

from ..config import Config
from .fairness import FairnessAnalyzer
from .metrics import MetricsCalculator


class ModelEvaluator:
    """Orchestrates comprehensive model evaluation."""

    def __init__(self, config: Config):
        self.config = config
        self.metrics_calc = MetricsCalculator(config.evaluation.threshold)
        self.fairness_analyzer = FairnessAnalyzer(config.evaluation.fairness_groups)

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Comprehensive evaluation of a single model.

        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test targets
            test_df: Optional full test DataFrame for fairness analysis

        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Evaluating model on {len(X_test)} test samples...")

        # Get predictions
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        y_test_values = np.asarray(y_test.values)

        # Calculate comprehensive metrics
        metrics = self.metrics_calc.calculate_all_metrics(y_test_values, y_proba)

        # Print metrics summary
        self.metrics_calc.print_metrics_summary(metrics)

        evaluation_results = {
            "metrics": metrics,
            "predictions": {
                "y_true": y_test.values.tolist(),
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
            },
        }

        # Fairness analysis if test_df is provided
        if test_df is not None:
            print("\nPerforming fairness analysis...")

            # Add predictions to test DataFrame
            test_df_with_preds = test_df.copy()
            test_df_with_preds["y_pred_proba"] = y_proba

            fairness_analysis = self.fairness_analyzer.analyze_all_groups(
                test_df_with_preds, "responded_within_2h", "y_pred_proba"
            )

            self.fairness_analyzer.print_fairness_summary(fairness_analysis)
            evaluation_results["fairness"] = fairness_analysis

        return evaluation_results

    def stress_test_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame
    ) -> Dict:
        """
        Perform stress testing on the model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            test_df: Full test DataFrame

        Returns:
            Dictionary containing stress test results
        """
        print("=" * 60)
        print("STRESS TESTING")
        print("=" * 60)

        stress_results = {}

        # 1. Missing data stress test
        print("\n1. Missing Data Stress Test:")
        X_missing = X_test.copy()

        # Randomly drop some features
        np.random.seed(self.config.env.seed)
        n_features_to_drop = int(
            len(X_test.columns) * self.config.evaluation.drop_fraction
        )
        features_to_drop = np.random.choice(
            X_test.columns, n_features_to_drop, replace=False
        )

        y_test_values = np.asarray(y_test.values)

        for col in features_to_drop:
            if col in X_missing.columns:
                X_missing[col] = np.nan

        # Fill NaN with median/mode (with safety checks)
        for col in X_missing.columns:
            if X_missing[col].dtype in ["float64", "int64"]:
                # Use median, but fallback to 0 if all values are NaN
                median_val = X_missing[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_missing[col] = X_missing[col].fillna(median_val)
            else:
                # Use mode, but fallback to "unknown" if no mode exists
                mode_values = X_missing[col].mode()
                fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "unknown"
                X_missing[col] = X_missing[col].fillna(fill_value)

        try:
            y_proba_missing = model.predict_proba(X_missing)

            metrics_missing = self.metrics_calc.calculate_all_metrics(
                y_test_values, y_proba_missing
            )

            print(
                f"  Original AUC: {self.metrics_calc.calculate_all_metrics(y_test_values, model.predict_proba(X_test))['auc']:.4f}"
            )
            print(f"  Missing Data AUC: {metrics_missing['auc']:.4f}")
            print(
                f"  AUC Drop: {self.metrics_calc.calculate_all_metrics(y_test_values, model.predict_proba(X_test))['auc'] - metrics_missing['auc']:.4f}"
            )

            stress_results["missing_data"] = {
                "metrics": metrics_missing,
                "features_dropped": features_to_drop.tolist(),
                "auc_drop": self.metrics_calc.calculate_all_metrics(
                    y_test_values, model.predict_proba(X_test)
                )["auc"]
                - metrics_missing["auc"],
            }
        except Exception as e:
            print(f"  Missing data test failed: {str(e)}")
            stress_results["missing_data"] = {"error": str(e)}

        # 2. Label noise stress test
        print("\n2. Label Noise Stress Test:")
        y_noisy = y_test.copy()

        # Add noise to labels
        n_flip = int(len(y_test) * self.config.evaluation.label_noise_fraction)
        flip_indices = np.random.choice(len(y_test), n_flip, replace=False)
        y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]

        try:
            y_proba_original = model.predict_proba(X_test)
            metrics_noisy = self.metrics_calc.calculate_all_metrics(
                np.asarray(y_noisy.values), y_proba_original
            )

            print(
                f"  Original AUC: {self.metrics_calc.calculate_all_metrics(y_test_values, y_proba_original)['auc']:.4f}"
            )
            print(f"  Noisy Labels AUC: {metrics_noisy['auc']:.4f}")
            print(
                f"  AUC Drop: {self.metrics_calc.calculate_all_metrics(y_test_values, y_proba_original)['auc'] - metrics_noisy['auc']:.4f}"
            )

            stress_results["label_noise"] = {
                "metrics": metrics_noisy,
                "noise_fraction": self.config.evaluation.label_noise_fraction,
                "auc_drop": self.metrics_calc.calculate_all_metrics(
                    y_test_values, y_proba_original
                )["auc"]
                - metrics_noisy["auc"],
            }
        except Exception as e:
            print(f"  Label noise test failed: {str(e)}")
            stress_results["label_noise"] = {"error": str(e)}

        return stress_results

    def compare_models(self, models_results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple models' evaluation results.

        Args:
            models_results: Dictionary mapping model names to their evaluation results

        Returns:
            Dictionary containing model comparison
        """
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison = {
            "model_rankings": {},
            "metric_comparison": {},
            "overall_ranking": [],
        }

        # Extract key metrics for comparison
        key_metrics = ["accuracy", "precision", "recall", "f1", "auc"]

        for metric in key_metrics:
            metric_values = {}
            for model_name, results in models_results.items():
                if "metrics" in results:
                    metric_values[model_name] = results["metrics"].get(metric, 0)

            # Rank models by this metric
            ranked_models = sorted(
                metric_values.items(), key=lambda x: x[1], reverse=True
            )
            comparison["metric_comparison"][metric] = metric_values
            comparison["model_rankings"][metric] = [model[0] for model in ranked_models]

        # Print comparison
        print("\nMetric Comparison:")
        print("-" * 40)
        for metric in key_metrics:
            print(f"\n{metric.upper()}:")
            for model_name, value in comparison["metric_comparison"][metric].items():
                print(f"  {model_name}: {value:.4f}")

        # Overall ranking (average rank across metrics)
        model_names = list(models_results.keys())
        avg_ranks = {}

        for model_name in model_names:
            ranks = []
            for metric in key_metrics:
                if metric in comparison["model_rankings"]:
                    try:
                        rank = (
                            comparison["model_rankings"][metric].index(model_name) + 1
                        )
                        ranks.append(rank)
                    except ValueError:
                        ranks.append(len(model_names))  # Worst rank if not found
            avg_ranks[model_name] = np.mean(ranks) if ranks else len(model_names)

        overall_ranking = sorted(avg_ranks.items(), key=lambda x: x[1])
        comparison["overall_ranking"] = [model[0] for model in overall_ranking]

        print("\nOverall Ranking (by average rank):")
        print("-" * 40)
        for i, (model_name, avg_rank) in enumerate(overall_ranking):
            print(f"  {i + 1}. {model_name} (avg rank: {avg_rank:.2f})")

        return comparison

    def save_evaluation_results(self, results: Dict, path: Path) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            path: Path to save the results
        """

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy(results)

        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Evaluation results saved to {path}")

    def evaluate_all_models(
        self,
        models: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
    ) -> Dict:
        """
        Evaluate all models comprehensively.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            test_df: Full test DataFrame

        Returns:
            Dictionary containing all evaluation results
        """
        all_results = {}

        metrics_dir = self.config.paths.metrics_dir or (
            self.config.paths.base_dir / "metrics"
        )

        for model_name, model in models.items():
            if model is not None:
                print(f"\n{'=' * 60}")
                print(f"EVALUATING {model_name.upper()} MODEL")
                print(f"{'=' * 60}")

                # Standard evaluation
                eval_results = self.evaluate_model(model, X_test, y_test, test_df)

                # Stress testing
                stress_results = self.stress_test_model(model, X_test, y_test, test_df)
                eval_results["stress_tests"] = stress_results

                all_results[model_name] = eval_results

                # Save individual model results
                model_results_path = metrics_dir / f"{model_name}_evaluation.json"
                self.save_evaluation_results(eval_results, model_results_path)

        # Compare models if multiple models evaluated
        if len(all_results) > 1:
            comparison = self.compare_models(all_results)
            all_results["model_comparison"] = comparison

        # Save combined results
        combined_results_path = metrics_dir / "all_models_evaluation.json"
        self.save_evaluation_results(all_results, combined_results_path)

        return all_results
