"""
Baseline model implementation for AI Medication Reminder.

Implements Logistic Regression baseline according to pipeline specification A.5.
"""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ModelConfig


class BaselineModel:
    """Baseline Logistic Regression model for medication reminder response prediction."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.

        Args:
            X: Feature DataFrame

        Returns:
            ColumnTransformer for preprocessing
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Create preprocessing steps
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import OneHotEncoder

        # Create preprocessing pipelines for numeric and categorical features
        numeric_pipeline = SkPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = SkPipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
                if categorical_features
                else ("cat", "passthrough", []),
            ],
            remainder="passthrough",
        )

        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """
        Train the baseline model.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Self for method chaining
        """
        print(
            f"Training baseline model with {len(X)} samples and {len(X.columns)} features"
        )

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)

        # Create model pipeline
        self.model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

        print("Baseline model training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from logistic regression coefficients.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        # Get coefficients
        coef = self.model.named_steps["classifier"].coef_[0]

        # Map to feature names (this is simplified - in practice you'd need to handle preprocessing)
        importance = dict(zip(self.feature_names, np.abs(coef)))

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": self.config,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Baseline model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        """
        Load a trained model.

        Args:
            path: Path to the saved model

        Returns:
            Loaded BaselineModel instance
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(model_data["config"])
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = True

        print(f"Baseline model loaded from {path}")
        return instance
