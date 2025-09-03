"""
Data validation module for AI Medication Reminder.

Validates data schema and constraints according to pipeline specification A.2.
"""

from typing import List, Tuple

import pandas as pd

from ..config import DataConfig


class DataValidator:
    """Validates medication reminder data according to schema requirements."""

    def __init__(self, config: DataConfig):
        self.config = config

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required columns and basic structure.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required columns
        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check if DataFrame is empty
        if len(df) == 0:
            errors.append("DataFrame is empty")

        return len(errors) == 0, errors

    def validate_constraints(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data constraints according to A.2 specification.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate channels
        invalid_channels = set(df["channel"]) - set(self.config.channels)
        if invalid_channels:
            errors.append(f"Invalid channels found: {invalid_channels}")

        # Validate responded_within_2h and ack_latency_sec constraint
        invalid_ack = df[
            (df["responded_within_2h"] == 0) & (df["ack_latency_sec"].notna())
        ]
        if len(invalid_ack) > 0:
            errors.append(
                f"Found {len(invalid_ack)} rows where responded_within_2h=0 but ack_latency_sec is not null"
            )

        # Validate timezone format
        tz_pattern = r"^[+-]\d{2}:\d{2}$"
        invalid_tz = df[~df["tz_offset"].str.match(tz_pattern, na=False)]
        if len(invalid_tz) > 0:
            errors.append(f"Found {len(invalid_tz)} rows with invalid timezone format")

        # Validate timestamp format
        try:
            pd.to_datetime(
                df["ts_reminder"], format="%Y-%m-%dT%H:%M:%SZ", errors="raise"
            )
        except Exception as e:
            errors.append(f"Invalid timestamp format: {str(e)}")

        return len(errors) == 0, errors

    def validate_temporal_ordering(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that timestamps are roughly ordered within each user.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        df_sorted = df.copy()
        df_sorted["_ts"] = pd.to_datetime(df_sorted["ts_reminder"])

        for user_pid, user_df in df_sorted.groupby("user_pid"):
            user_df = user_df.sort_values("_ts")
            if not user_df["_ts"].is_monotonic_increasing:
                warnings.append(f"User {user_pid} has non-monotonic timestamps")

        return len(warnings) == 0, warnings

    def validate_full(self, df: pd.DataFrame) -> Tuple[bool, dict]:
        """
        Run full validation suite.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "schema": {"valid": False, "errors": []},
            "constraints": {"valid": False, "errors": []},
            "temporal": {"valid": False, "warnings": []},
        }

        # Schema validation
        schema_valid, schema_errors = self.validate_schema(df)
        report["schema"] = {"valid": schema_valid, "errors": schema_errors}

        if not schema_valid:
            return False, report

        # Constraints validation
        constraints_valid, constraint_errors = self.validate_constraints(df)
        report["constraints"] = {
            "valid": constraints_valid,
            "errors": constraint_errors,
        }

        # Temporal validation (warnings only)
        temporal_valid, temporal_warnings = self.validate_temporal_ordering(df)
        report["temporal"] = {"valid": temporal_valid, "warnings": temporal_warnings}

        overall_valid = schema_valid and constraints_valid
        return overall_valid, report
