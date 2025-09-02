"""
Behavioral feature engineering for AI Medication Reminder.

Creates user behavior-based features according to pipeline specification A.4.
"""

import warnings
from typing import List

import numpy as np
import pandas as pd

# Suppress pandas FutureWarning about groupby behavior
warnings.filterwarnings(
    "ignore", message=".*groupby.*deprecated.*", category=FutureWarning
)


class BehavioralFeatures:
    """Creates behavioral features from user interaction history."""

    def __init__(self, rolling_windows: List[int] = [7, 14]):
        self.rolling_windows = rolling_windows

    def create_lag_features(
        self, df: pd.DataFrame, lags: List[int] = [1, 2]
    ) -> pd.DataFrame:
        """
        Create lag features for previous responses.

        Args:
            df: DataFrame sorted by user_pid and _ts
            lags: List of lag periods to create

        Returns:
            DataFrame with lag_1, lag_2, etc. columns
        """
        df = df.copy()

        for lag in lags:
            df[f"lag_{lag}"] = df.groupby("user_pid")["responded_within_2h"].shift(lag)

        return df

    def create_rolling_ctr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling click-through rate (CTR) features.

        Args:
            df: DataFrame sorted by user_pid and _ts

        Returns:
            DataFrame with ctr7, ctr14, etc. columns
        """
        df = df.copy()

        def rolling_ctr(series, window):
            """Calculate rolling CTR excluding current observation."""
            return series.shift(1).rolling(window=window, min_periods=1).mean()

        for window in self.rolling_windows:
            df[f"ctr{window}"] = df.groupby("user_pid")[
                "responded_within_2h"
            ].transform(lambda s: rolling_ctr(s, window))

        return df

    def create_channel_ctr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-user per-channel CTR features.

        Args:
            df: DataFrame with user_pid, channel, responded_within_2h

        Returns:
            DataFrame with ctr_user_channel column
        """
        df = df.copy()

        # Calculate historical CTR for each user-channel combination
        # Exclude current observation to avoid data leakage
        # Use transform instead of apply to avoid FutureWarning
        df = df.sort_values(["user_pid", "channel", "ts_reminder"]).reset_index(
            drop=True
        )
        df["ctr_user_channel"] = df.groupby(["user_pid", "channel"])[
            "responded_within_2h"
        ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

        # Fill NaN values with overall user CTR
        overall_ctr = df.groupby("user_pid")["responded_within_2h"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        df["ctr_user_channel"] = df["ctr_user_channel"].fillna(overall_ctr)

        return df

    def create_ack_latency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create acknowledgment latency features.

        Args:
            df: DataFrame with ack_latency_sec column

        Returns:
            DataFrame with processed ack_latency_sec and latency_bucket
        """
        df = df.copy()

        # Fill NaN values with median
        median_latency = df["ack_latency_sec"].median()
        df["ack_latency_sec"] = df["ack_latency_sec"].fillna(median_latency)

        # Create latency buckets
        def latency_bucket(latency):
            if pd.isna(latency):
                return "none"
            elif latency <= 300:  # 5 minutes
                return "fast"
            elif latency <= 1800:  # 30 minutes
                return "medium"
            else:
                return "slow"

        df["latency_bucket"] = df["ack_latency_sec"].apply(latency_bucket)

        return df

    def create_exponential_decay_features(
        self, df: pd.DataFrame, lambda_decay: float = 0.9
    ) -> pd.DataFrame:
        """
        Create exponentially decayed response history features.

        Args:
            df: DataFrame sorted by user_pid and _ts
            lambda_decay: Decay factor (0 < lambda < 1)

        Returns:
            DataFrame with exp_decay_response column
        """
        df = df.copy()

        def exp_decay_sum(series, lam):
            """Calculate exponentially decayed sum of past responses."""
            result = np.zeros(len(series))
            for i in range(1, len(series)):
                # Sum of lambda^j * response[i-j-1] for j from 1 to i
                decay_sum = 0
                for j in range(1, i + 1):
                    if i - j >= 0:
                        decay_sum += (lam**j) * series.iloc[i - j]
                result[i] = decay_sum
            return pd.Series(result, index=series.index)

        df["exp_decay_response"] = df.groupby("user_pid")[
            "responded_within_2h"
        ].transform(lambda s: exp_decay_sum(s, lambda_decay))

        return df

    def create_all_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all behavioral features.

        Args:
            df: DataFrame sorted by user_pid and _ts

        Returns:
            DataFrame with all behavioral features
        """
        df = self.create_lag_features(df)
        df = self.create_rolling_ctr(df)
        df = self.create_channel_ctr(df)
        df = self.create_ack_latency_features(df)
        df = self.create_exponential_decay_features(df)

        return df

    def get_behavioral_feature_names(self) -> List[str]:
        """Get list of behavioral feature column names."""
        features = [
            "lag_1",
            "lag_2",
            "ack_latency_sec",
            "latency_bucket",
            "ctr_user_channel",
            "exp_decay_response",
        ]

        # Add rolling CTR features
        for window in self.rolling_windows:
            features.append(f"ctr{window}")

        return features
