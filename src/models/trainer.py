"""
Model training orchestrator for AI Medication Reminder.

Handles training of both baseline and advanced models according to pipeline specification A.5.
"""

from typing import Dict, Optional, Union

import pandas as pd

from ..config import Config
from .baseline import BaselineModel
from .tiny_temporal import TinyTemporalModel


class ModelTrainer:
    """Orchestrates training of different model types."""

    def __init__(self, config: Config):
        self.config = config
        self.models = {}

    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> BaselineModel:
        """
        Train baseline logistic regression model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Trained BaselineModel
        """
        print("=" * 50)
        print("Training Baseline Model (Logistic Regression)")
        print("=" * 50)

        # Create and train model
        model = BaselineModel(self.config.model)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_proba = model.predict_proba(X_val)
        val_pred = model.predict(X_val)

        # Calculate basic metrics
        from sklearn.metrics import accuracy_score, roc_auc_score

        val_accuracy = accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")

        # Save model
        model_path = self.config.paths.models_dir / "baseline_model.pkl"
        model.save(model_path)

        # Store in trainer
        self.models["baseline"] = model

        return model

    def train_tiny_temporal(
        self,
        X_seq_train: pd.DataFrame,
        y_seq_train: pd.Series,
        X_seq_val: pd.DataFrame,
        y_seq_val: pd.Series,
    ) -> Optional[TinyTemporalModel]:
        """
        Train TinyTemporal model if PyTorch is available and enabled.

        Args:
            X_seq_train: Training sequences
            y_seq_train: Training targets
            X_seq_val: Validation sequences
            y_seq_val: Validation targets

        Returns:
            Trained TinyTemporalModel or None if not available
        """
        if not self.config.model.use_tiny_temporal:
            print("TinyTemporal training skipped (not enabled in config)")
            return None

        print("=" * 50)
        print("Training TinyTemporal Model")
        print("=" * 50)

        try:
            # Create and train model
            model = TinyTemporalModel(self.config.model)

            if not model.torch_available:
                print("PyTorch not available. Skipping TinyTemporal training.")
                return None

            model.fit(X_seq_train, y_seq_train)

            # Evaluate on validation set
            val_proba = model.predict_proba(X_seq_val)
            val_pred = model.predict(X_seq_val)

            # Calculate basic metrics
            from sklearn.metrics import accuracy_score, roc_auc_score

            val_accuracy = accuracy_score(y_seq_val, val_pred)
            val_auc = roc_auc_score(y_seq_val, val_proba)

            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation AUC: {val_auc:.4f}")

            # Save model
            model_path = self.config.paths.models_dir / "tiny_temporal_model.pth"
            model.save(model_path)

            # Store in trainer
            self.models["tiny_temporal"] = model

            return model

        except Exception as e:
            print(f"TinyTemporal training failed: {str(e)}")
            return None

    def train_all_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_engineer
    ) -> Dict[str, Union[BaselineModel, TinyTemporalModel]]:
        """
        Train all available models.

        Args:
            train_df: Training DataFrame with features
            val_df: Validation DataFrame with features
            feature_engineer: FeatureEngineer instance

        Returns:
            Dictionary of trained models
        """
        print("Starting model training pipeline...")

        # Prepare standard features for baseline model
        X_train, y_train = feature_engineer.prepare_model_features(train_df)
        X_val, y_val = feature_engineer.prepare_model_features(val_df)

        # Train baseline model
        baseline_model = self.train_baseline(X_train, y_train, X_val, y_val)

        # Prepare sequences for temporal model
        if self.config.model.use_tiny_temporal:
            print("\nPreparing sequences for TinyTemporal model...")
            X_seq_train, y_seq_train = (
                feature_engineer.create_sequences_for_temporal_model(train_df)
            )
            X_seq_val, y_seq_val = feature_engineer.create_sequences_for_temporal_model(
                val_df
            )

            if len(X_seq_train) > 0:
                tiny_temporal_model = self.train_tiny_temporal(
                    X_seq_train, y_seq_train, X_seq_val, y_seq_val
                )
            else:
                print("No sequences available for TinyTemporal training")

        print("\nModel training pipeline completed!")
        return self.models

    def get_model(
        self, model_name: str
    ) -> Optional[Union[BaselineModel, TinyTemporalModel]]:
        """
        Get a trained model by name.

        Args:
            model_name: Name of the model ('baseline' or 'tiny_temporal')

        Returns:
            Trained model or None if not found
        """
        return self.models.get(model_name)

    def load_models(self) -> Dict[str, Union[BaselineModel, TinyTemporalModel]]:
        """
        Load previously saved models.

        Returns:
            Dictionary of loaded models
        """
        loaded_models = {}

        # Try to load baseline model
        baseline_path = self.config.paths.models_dir / "baseline_model.pkl"
        if baseline_path.exists():
            try:
                baseline_model = BaselineModel.load(baseline_path)
                loaded_models["baseline"] = baseline_model
                print("Baseline model loaded successfully")
            except Exception as e:
                print(f"Failed to load baseline model: {str(e)}")

        # Try to load TinyTemporal model
        tiny_temporal_path = self.config.paths.models_dir / "tiny_temporal_model.pth"
        if tiny_temporal_path.exists():
            try:
                tiny_temporal_model = TinyTemporalModel.load(tiny_temporal_path)
                loaded_models["tiny_temporal"] = tiny_temporal_model
                print("TinyTemporal model loaded successfully")
            except Exception as e:
                print(f"Failed to load TinyTemporal model: {str(e)}")

        self.models.update(loaded_models)
        return loaded_models
