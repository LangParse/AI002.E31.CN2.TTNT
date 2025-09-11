"""
Data processing module for AI Medication Reminder.

Handles data loading, preprocessing, and basic transformations.
"""

from typing import Tuple

import pandas as pd

from ..config import Config
from .generator import SyntheticDataGenerator
from .validator import DataValidator


class DataProcessor:
    """Main data processing class for medication reminder data."""

    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator(config.data)
        self.generator = SyntheticDataGenerator(config.data, config.env.seed)

    def load_or_generate_data(self, force_generate: bool = False) -> pd.DataFrame:
        """
        Load existing data or generate synthetic data if not available.

        Args:
            force_generate: If True, generate new synthetic data even if file exists

        Returns:
            DataFrame with medication reminder data
        """
        data_dir = self.config.paths.data_dir or (self.config.paths.base_dir / "data")
        logs_path = data_dir / "logs_small.csv"

        if not logs_path.exists() or force_generate:
            print(f"Generating synthetic data (scale: {self.config.env.data_scale})")
            df = self.generator.save_synthetic_data(
                logs_path, self.config.env.data_scale
            )
        else:
            print(f"Loading existing data from {logs_path}")
            df = pd.read_csv(logs_path)

        # Validate data
        is_valid, report = self.validator.validate_full(df)
        if not is_valid:
            print("Data validation failed:")
            for section, details in report.items():
                if not details["valid"] and details.get("errors"):
                    print(f"  {section}: {details['errors']}")
            raise ValueError("Data validation failed")

        print(f"Loaded {len(df)} rows for {df['user_pid'].nunique()} users")
        return df

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse UTC timestamps and add timezone-aware datetime column.

        Args:
            df: DataFrame with ts_reminder column

        Returns:
            DataFrame with additional _ts column
        """
        df = df.copy()
        df["_ts"] = pd.to_datetime(
            df["ts_reminder"], format="%Y-%m-%dT%H:%M:%SZ", utc=True
        )
        return df

    def sort_by_user_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame by user_pid and timestamp.

        Args:
            df: DataFrame to sort

        Returns:
            Sorted DataFrame
        """
        if "_ts" not in df.columns:
            df = self.parse_timestamps(df)
        return df.sort_values(["user_pid", "_ts"]).reset_index(drop=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Parse timestamps if not already done
        if "_ts" not in df.columns:
            df = self.parse_timestamps(df)

        # Sort by user and time
        df = self.sort_by_user_time(df)

        # Convert numeric columns
        df["ack_latency_sec"] = pd.to_numeric(df["ack_latency_sec"], errors="coerce")
        df["responded_within_2h"] = df["responded_within_2h"].astype(int)
        df["delivered"] = df["delivered"].astype(int)

        return df

    def split_users(
        self,
        df: pd.DataFrame,
        test_size: float = 0.4,
        val_split: float = 0.5,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by users to avoid data leakage.

        Args:
            df: DataFrame to split
            test_size: Fraction of users for test+validation
            val_split: Fraction of test+val users for validation
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        users = df["user_pid"].unique()

        # Split users into train and temp (test+val)
        users_train, users_temp = train_test_split(
            users, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Split temp into val and test
        users_val, users_test = train_test_split(
            users_temp, test_size=val_split, random_state=random_state, shuffle=True
        )

        # Create DataFrames
        train_df = df[df["user_pid"].isin(users_train)].copy()
        val_df = df[df["user_pid"].isin(users_val)].copy()
        test_df = df[df["user_pid"].isin(users_test)].copy()

        print(
            f"Data split: Train={len(train_df)} ({len(users_train)} users), "
            f"Val={len(val_df)} ({len(users_val)} users), "
            f"Test={len(test_df)} ({len(users_test)} users)"
        )

        return train_df, val_df, test_df

    def prepare_data(
        self, force_generate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline.

        Args:
            force_generate: Whether to force generate new synthetic data

        Returns:
            Tuple of (full_df, train_df, val_df, test_df)
        """
        # Load or generate data
        df = self.load_or_generate_data(force_generate)

        # Clean data
        df = self.clean_data(df)

        # Split data
        train_df, val_df, test_df = self.split_users(
            df,
            test_size=self.config.model.test_size,
            val_split=self.config.model.val_split,
            random_state=self.config.env.seed,
        )

        return df, train_df, val_df, test_df
