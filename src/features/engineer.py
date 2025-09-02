"""
Main feature engineering orchestrator for AI Medication Reminder.

Combines temporal and behavioral feature engineering according to pipeline specification A.4.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .behavioral import BehavioralFeatures
from .temporal import TemporalFeatures


class FeatureEngineer:
    """Main feature engineering class that orchestrates all feature creation."""

    def __init__(self, config: Config):
        self.config = config
        self.temporal = TemporalFeatures()
        self.behavioral = BehavioralFeatures(config.model.rolling_windows)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data.

        Args:
            df: DataFrame with cleaned data (must have _ts column and be sorted)

        Returns:
            DataFrame with all engineered features
        """
        print("Creating temporal features...")
        df = self.temporal.create_all_temporal_features(df)

        print("Creating behavioral features...")
        df = self.behavioral.create_all_behavioral_features(df)

        # Drop rows with NaN in lag features (first few rows per user)
        initial_rows = len(df)
        df = df.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to missing lag features")

        # Additional data cleaning - fill remaining NaN values
        # Fill numeric NaN with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled {col} NaN values with median: {median_val:.3f}")

        # Fill categorical NaN with mode or 'unknown'
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "unknown"
                df[col] = df[col].fillna(fill_val)
                print(f"Filled {col} NaN values with: {fill_val}")

        # Final check for any remaining NaN
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            print(f"Warning: {remaining_nan} NaN values still remain after cleaning")
            # Show which columns still have NaN
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"Columns with NaN: {nan_cols}")

        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def get_feature_columns(self) -> dict:
        """
        Get categorized feature column names.

        Returns:
            Dictionary with feature categories and their column names
        """
        temporal_features = self.temporal.get_temporal_feature_names()
        behavioral_features = self.behavioral.get_behavioral_feature_names()

        # Separate numeric and categorical features
        numeric_features = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "hours_since_prev",
            "lag_1",
            "lag_2",
            "ack_latency_sec",
            "ctr_user_channel",
            "exp_decay_response",
        ]

        # Add rolling CTR features
        for window in self.config.model.rolling_windows:
            numeric_features.append(f"ctr{window}")

        categorical_features = [
            "channel",
            "time_bucket",
            "is_weekend",
            "latency_bucket",
        ]

        return {
            "temporal": temporal_features,
            "behavioral": behavioral_features,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "all": temporal_features + behavioral_features,
        }

    def save_feature_schema(self, path: Path) -> None:
        """
        Save feature schema to JSON file.

        Args:
            path: Path to save the schema file
        """
        schema = self.get_feature_columns()
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Feature schema saved to {path}")

    def prepare_model_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training.

        Args:
            df: DataFrame with all features

        Returns:
            Tuple of (X, y) where X contains features and y contains target
        """
        feature_cols = self.get_feature_columns()

        # Select feature columns
        X = df[feature_cols["numeric"] + feature_cols["categorical"]].copy()

        # Target variable
        y = df["responded_within_2h"].astype(int)

        return X, y

    def create_sequences_for_temporal_model(
        self, df: pd.DataFrame, window_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sequences for temporal models like TinyTemporal.

        Args:
            df: DataFrame with features
            window_size: Size of the sequence window

        Returns:
            Tuple of (X_sequences, y_sequences) for temporal modeling
        """
        if window_size is None:
            window_size = self.config.model.window_size

        feature_cols = self.get_feature_columns()["numeric"]

        sequences_X = []
        sequences_y = []

        for user_pid, user_df in df.groupby("user_pid"):
            user_df = user_df.sort_values("_ts")

            # Extract numeric features and target
            X_user = user_df[feature_cols].values
            y_user = user_df["responded_within_2h"].values

            # Create sequences
            for i in range(window_size, len(user_df)):
                sequences_X.append(X_user[i - window_size : i])
                sequences_y.append(y_user[i])

        if len(sequences_X) == 0:
            print("Warning: No sequences created. Data might be too small.")
            return pd.DataFrame(), pd.DataFrame()

        # Convert to DataFrames for consistency
        X_seq = pd.DataFrame(
            {
                "sequences": sequences_X,
                "feature_names": [feature_cols] * len(sequences_X),
            }
        )
        y_seq = pd.DataFrame({"target": sequences_y})

        print(f"Created {len(sequences_X)} sequences of length {window_size}")
        return X_seq, y_seq
