"""
Main pipeline orchestrator for AI Medication Reminder.

Orchestrates the complete pipeline from data processing to model evaluation and bandit simulation.
"""

import time
from typing import Dict, Optional

from .bandit import BanditSimulator
from .config import Config
from .data import DataProcessor
from .evaluation import ModelEvaluator
from .features import FeatureEngineer
from .models import ModelTrainer
from .utils import format_duration, print_section_header, setup_logging


class Pipeline:
    """Main pipeline orchestrator for the AI Medication Reminder system."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging("INFO", self.config.paths.base_dir / "pipeline.log")

        # Initialize components
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        self.model_evaluator = ModelEvaluator(config)
        self.bandit_simulator = BanditSimulator(config)

        # Storage for pipeline results
        self.results = {}

    def run_full_pipeline(self, force_retrain: bool = False) -> Dict:
        """
        Run the complete pipeline from data processing to evaluation.

        Args:
            force_retrain: Whether to force retraining of models

        Returns:
            Dictionary containing all pipeline results
        """
        start_time = time.time()

        print_section_header("AI MEDICATION REMINDER PIPELINE")
        print(f"Environment: {'Colab' if self.config.env.in_colab else 'Local'}")
        print(f"Data Scale: {self.config.env.data_scale}")
        print(f"GPU Available: {self.config.env.has_gpu}")
        print()

        try:
            # Step 1: Data Processing (A.1, A.2, A.6)
            print_section_header("STEP 1: DATA PROCESSING")
            data_results = self._run_data_processing()
            self.results["data"] = data_results

            # Step 2: Feature Engineering (A.4)
            print_section_header("STEP 2: FEATURE ENGINEERING")
            feature_results = self._run_feature_engineering(data_results)
            self.results["features"] = feature_results

            # Step 3: Model Training (A.5)
            print_section_header("STEP 3: MODEL TRAINING")
            model_results = self._run_model_training(feature_results, force_retrain)
            self.results["models"] = model_results

            # Step 4: Model Evaluation (A.3)
            print_section_header("STEP 4: MODEL EVALUATION")
            evaluation_results = self._run_model_evaluation(
                model_results, feature_results
            )
            self.results["evaluation"] = evaluation_results

            # Step 5: Bandit Simulation (A.7)
            print_section_header("STEP 5: BANDIT SIMULATION")
            bandit_results = self._run_bandit_simulation(feature_results)
            self.results["bandit"] = bandit_results

            # Pipeline Summary
            end_time = time.time()
            duration = end_time - start_time

            print_section_header("PIPELINE COMPLETED")
            print(f"Total Duration: {format_duration(duration)}")
            print(f"Results saved to: {self.config.paths.base_dir}")

            self.results["pipeline_info"] = {
                "duration_seconds": duration,
                "config": self.config.to_dict(),
                "timestamp": time.time(),
            }

            return self.results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            print(f"Pipeline failed: {str(e)}")
            raise

    def _run_data_processing(self) -> Dict:
        """Run data processing steps."""
        print("Loading and processing data...")

        # Load or generate data
        full_df, train_df, val_df, test_df = self.data_processor.prepare_data()

        print(
            f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        return {
            "full_df": full_df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "data_info": {
                "total_samples": len(full_df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "n_users": full_df["user_pid"].nunique(),
                "response_rate": full_df["responded_within_2h"].mean(),
            },
        }

    def _run_feature_engineering(self, data_results: Dict) -> Dict:
        """Run feature engineering steps."""
        print("Creating features...")

        # Create features for all datasets
        train_df_features = self.feature_engineer.create_all_features(
            data_results["train_df"]
        )
        val_df_features = self.feature_engineer.create_all_features(
            data_results["val_df"]
        )
        test_df_features = self.feature_engineer.create_all_features(
            data_results["test_df"]
        )

        # Save feature schema
        data_dir = self.config.paths.data_dir or (self.config.paths.base_dir / "data")
        schema_path = data_dir / "feature_schema.json"
        self.feature_engineer.save_feature_schema(schema_path)

        return {
            "train_df": train_df_features,
            "val_df": val_df_features,
            "test_df": test_df_features,
            "feature_info": {
                "n_features": len(self.feature_engineer.get_feature_columns()["all"]),
                "feature_schema": self.feature_engineer.get_feature_columns(),
            },
        }

    def _run_model_training(self, feature_results: Dict, force_retrain: bool) -> Dict:
        """Run model training steps."""
        if not force_retrain:
            # Try to load existing models
            try:
                existing_models = self.model_trainer.load_models()
                if existing_models:
                    print("Loaded existing models")
                    return {"models": existing_models, "training_skipped": True}
            except Exception as e:
                print(f"Failed to load existing models: {str(e)}")

        print("Training models...")

        # Train all models
        trained_models = self.model_trainer.train_all_models(
            feature_results["train_df"],
            feature_results["val_df"],
            self.feature_engineer,
        )

        return {"models": trained_models, "training_skipped": False}

    def _run_model_evaluation(self, model_results: Dict, feature_results: Dict) -> Dict:
        """Run model evaluation steps."""
        print("Evaluating models...")

        # Prepare test data
        X_test, y_test = self.feature_engineer.prepare_model_features(
            feature_results["test_df"]
        )

        # Evaluate all models
        evaluation_results = self.model_evaluator.evaluate_all_models(
            model_results["models"], X_test, y_test, feature_results["test_df"]
        )

        return evaluation_results

    def _run_bandit_simulation(self, feature_results: Dict) -> Dict:
        """Run bandit simulation steps."""
        print("Running bandit simulation...")

        # Get context features for bandit
        context_features = self.config.bandit.context_features

        # Run bandit comparison
        bandit_results = self.bandit_simulator.compare_policies(
            feature_results["test_df"], context_features
        )

        # Save bandit results
        metrics_dir = self.config.paths.metrics_dir or (
            self.config.paths.base_dir / "metrics"
        )
        bandit_results_path = metrics_dir / "bandit_results.json"
        from .utils import save_results

        save_results(bandit_results, bandit_results_path, "json")

        return bandit_results

    def run_inference(
        self, user_data: Dict, medications: Optional[list] = None
    ) -> Dict:
        """
        Run end-to-end inference for a single user.

        Args:
            user_data: Dictionary with user context features
            medications: Optional list of medications for DDI checking

        Returns:
            Dictionary with recommendations and warnings
        """
        print("Running inference...")

        # Load trained models if not already loaded
        if not hasattr(self, "_loaded_models"):
            self._loaded_models = self.model_trainer.load_models()

        # Drug interaction checking
        warnings = []
        if medications:
            from .utils import DrugInteractionChecker

            ddi_checker = DrugInteractionChecker()

            # Check drug interactions
            interactions = ddi_checker.check_drug_interactions(medications)
            if interactions:
                warnings.extend(interactions)

            # Check contraindications (simplified patient profile)
            patient_profile = {
                "age": user_data.get("age", 50),
                "conditions": user_data.get("conditions", []),
                "pregnancy_status": user_data.get("pregnancy_status"),
            }

            for med in medications:
                contraindications = ddi_checker.check_contraindications(
                    med, patient_profile
                )
                warnings.extend(contraindications)

        # Channel recommendation using bandit
        if "baseline" in self._loaded_models:
            model = self._loaded_models["baseline"]

            # Create feature vector (simplified)
            import numpy as np
            import pandas as pd

            # Mock feature creation for inference
            features = {
                "hour_sin": np.sin(2 * np.pi * user_data.get("hour", 12) / 24),
                "hour_cos": np.cos(2 * np.pi * user_data.get("hour", 12) / 24),
                "dow_sin": np.sin(2 * np.pi * user_data.get("dow", 1) / 7),
                "dow_cos": np.cos(2 * np.pi * user_data.get("dow", 1) / 7),
                "hours_since_prev": user_data.get("hours_since_prev", 24),
                "ctr7": user_data.get("ctr7", 0.5),
                "ctr14": user_data.get("ctr14", 0.5),
                "ack_latency_sec": user_data.get("ack_latency_sec", 300),
                "lag_1": user_data.get("lag_1", 0),
                "lag_2": user_data.get("lag_2", 0),
                "channel": user_data.get("channel", "push"),
                "time_bucket": user_data.get("time_bucket", "morning"),
                "is_weekend": user_data.get("is_weekend", 0),
                "latency_bucket": user_data.get("latency_bucket", "medium"),
            }

            # Convert to DataFrame
            X_inference = pd.DataFrame([features])

            try:
                # Get prediction
                response_prob = model.predict_proba(X_inference)[0]

                # Channel recommendation using epsilon-greedy bandit
                from .bandit import EpsilonGreedyBandit

                bandit = EpsilonGreedyBandit(self.config.bandit)
                recommended_channel = bandit.select_arm(features)

                recommendations = {
                    "recommended_channel": recommended_channel,
                    "response_probability": float(response_prob),
                    "confidence": "high"
                    if abs(response_prob - 0.5) > 0.2
                    else "medium",
                }
            except Exception as e:
                print(f"Inference failed: {str(e)}")
                recommendations = {
                    "recommended_channel": "push",  # Default
                    "response_probability": 0.5,
                    "confidence": "low",
                    "error": str(e),
                }
        else:
            recommendations = {
                "recommended_channel": "push",  # Default
                "response_probability": 0.5,
                "confidence": "low",
                "error": "No trained model available",
            }

        return {
            "recommendations": recommendations,
            "warnings": warnings,
            "user_context": user_data,
        }
