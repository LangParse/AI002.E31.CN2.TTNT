"""
Synthetic data generation module for AI Medication Reminder.

Generates synthetic medication reminder data according to pipeline specification A.6.
"""

import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig


class SyntheticDataGenerator:
    """Generates synthetic medication reminder data for testing and development."""

    def __init__(self, config: DataConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _get_scale_params(self, scale: str) -> Tuple[int, int]:
        """Get number of users and days based on scale."""
        if scale.upper() == "SMALL":
            n_users = self.rng.integers(*self.config.n_users_small)
            n_days = self.rng.integers(*self.config.n_days_small)
        else:  # LARGE
            n_users = self.rng.integers(*self.config.n_users_large)
            n_days = self.rng.integers(*self.config.n_days_large)
        return n_users, n_days

    def _generate_base_probability(
        self, channel: str, hour: int, is_weekend: bool
    ) -> float:
        """Generate base response probability based on channel, time, and day type."""
        # Base probability
        p = 0.55

        # Channel adjustment
        if channel == "push":
            p += 0.05
        elif channel == "voice":
            p -= 0.05

        # Time of day adjustment
        if 5 <= hour <= 11:  # Morning
            p += 0.10
        elif 18 <= hour <= 22:  # Evening
            p -= 0.05

        # Weekend adjustment
        if is_weekend:
            p -= 0.03

        return float(np.clip(p, 0.05, 0.95))

    def _select_channel(self, hour: int) -> str:
        """Select channel based on time of day with different probabilities."""
        if 5 <= hour <= 11:  # Morning
            return self.rng.choice(self.config.channels, p=[0.5, 0.3, 0.2])
        elif 12 <= hour <= 17:  # Afternoon
            return self.rng.choice(self.config.channels, p=[0.3, 0.5, 0.2])
        else:  # Evening/Night
            return self.rng.choice(self.config.channels, p=[0.3, 0.3, 0.4])

    def generate_synthetic_data(self, scale: str = "SMALL") -> pd.DataFrame:
        """
        Generate synthetic medication reminder data.

        Args:
            scale: "SMALL" for local testing, "LARGE" for Colab training

        Returns:
            DataFrame with synthetic medication reminder data
        """
        n_users, n_days = self._get_scale_params(scale)

        base_date = datetime.datetime(2025, 8, 1, 0, 0, tzinfo=datetime.timezone.utc)
        rows = []

        for u in range(1, n_users + 1):
            user_id = f"u_{u:02d}"

            for d in range(n_days):
                # Generate 2 reminders per day (morning and evening)
                for hour, minute in [(6, 30), (20, 30)]:
                    # Add jitter to timing
                    jitter_min = int(self.rng.integers(-25, 26))

                    ts = base_date + datetime.timedelta(
                        days=int(d), hours=hour, minutes=minute + jitter_min
                    )

                    # Select channel based on time
                    channel = self._select_channel(ts.hour)

                    # Generate response probability
                    is_weekend = ts.weekday() >= 5
                    p_respond = self._generate_base_probability(
                        channel, ts.hour, is_weekend
                    )

                    # Generate response
                    responded = int(self.rng.random() < p_respond)

                    # Generate acknowledgment latency if responded
                    ack_latency = (
                        int(self.rng.integers(120, 900)) if responded else np.nan
                    )

                    rows.append(
                        [
                            user_id,
                            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "+07:00",  # Fixed timezone for simplicity
                            channel,
                            1,  # Always delivered for synthetic data
                            responded,
                            ack_latency,
                            0,  # No snooze for synthetic data
                        ]
                    )

        df = pd.DataFrame(
            rows,
            columns=[
                "user_pid",
                "ts_reminder",
                "tz_offset",
                "channel",
                "delivered",
                "responded_within_2h",
                "ack_latency_sec",
                "snooze",
            ],
        )

        return df

    def save_synthetic_data(self, path: Path, scale: str = "SMALL") -> pd.DataFrame:
        """
        Generate and save synthetic data to CSV file.

        Args:
            path: Path to save the CSV file
            scale: Data scale ("SMALL" or "LARGE")

        Returns:
            Generated DataFrame
        """
        df = self.generate_synthetic_data(scale)
        df.to_csv(path, index=False)
        return df
