"""
Temporal feature engineering for AI Medication Reminder.

Creates time-based features according to pipeline specification A.4.
"""

from typing import List

import numpy as np
import pandas as pd


class TemporalFeatures:
    """Creates temporal features from timestamp data."""

    def __init__(self):
        pass

    def create_time_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical time encodings for hour and day of week.

        Args:
            df: DataFrame with _ts column

        Returns:
            DataFrame with additional time encoding columns
        """
        df = df.copy()

        # Extract hour with minutes as decimal
        df["hour"] = df["_ts"].dt.hour + df["_ts"].dt.minute / 60.0

        # Cyclical encoding for hour (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week (0=Monday, 6=Sunday)
        df["dow"] = df["_ts"].dt.weekday

        # Cyclical encoding for day of week (7-day cycle)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

        return df

    def create_time_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time bucket categories.

        Args:
            df: DataFrame with hour column

        Returns:
            DataFrame with time_bucket column
        """
        df = df.copy()

        def time_bucket(h):
            h = int(h)
            if 5 <= h <= 11:
                return "morning"
            elif 12 <= h <= 17:
                return "afternoon"
            elif 18 <= h <= 22:
                return "evening"
            else:
                return "other"

        df["time_bucket"] = df["hour"].apply(time_bucket)
        return df

    def create_weekend_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weekend indicator.

        Args:
            df: DataFrame with _ts column

        Returns:
            DataFrame with is_weekend column
        """
        df = df.copy()
        df["is_weekend"] = df["_ts"].dt.weekday.isin([5, 6]).astype(int)
        return df

    def create_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time gap features between consecutive reminders.

        Args:
            df: DataFrame sorted by user_pid and _ts

        Returns:
            DataFrame with hours_since_prev column
        """
        df = df.copy()

        # Calculate time difference in hours
        df["hours_since_prev"] = (
            df.groupby("user_pid")["_ts"].diff().dt.total_seconds() / 3600.0
        )

        # Fill NaN values (first reminder for each user) with median
        median_gap = df["hours_since_prev"].median()
        df["hours_since_prev"] = df["hours_since_prev"].fillna(median_gap)

        return df

    def create_all_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all temporal features.

        Args:
            df: DataFrame with _ts column, sorted by user_pid and _ts

        Returns:
            DataFrame with all temporal features
        """
        df = self.create_time_encodings(df)
        df = self.create_time_buckets(df)
        df = self.create_weekend_indicator(df)
        df = self.create_time_gaps(df)

        return df

    def get_temporal_feature_names(self) -> List[str]:
        """Get list of temporal feature column names."""
        return [
            "hour",
            "hour_sin",
            "hour_cos",
            "dow",
            "dow_sin",
            "dow_cos",
            "time_bucket",
            "is_weekend",
            "hours_since_prev",
        ]
