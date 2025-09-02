"""
Configuration management for AI Medication Reminder system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""

    in_colab: bool = field(
        default_factory=lambda: "google.colab" in __import__("sys").modules
    )
    has_gpu: bool = False
    data_scale: str = field(
        default_factory=lambda: os.environ.get(
            "DATA_SCALE",
            "LARGE" if "google.colab" in __import__("sys").modules else "SMALL",
        ).upper()
    )
    use_gdrive: bool = False
    seed: int = 42

    def __post_init__(self):
        """Validate and set derived values."""
        assert self.data_scale in {"SMALL", "LARGE"}, (
            f"data_scale must be SMALL or LARGE, got {self.data_scale}"
        )

        # Check GPU availability
        if self.in_colab:
            try:
                import torch  # type: ignore

                self.has_gpu = torch.cuda.is_available()
            except ImportError:
                self.has_gpu = False


@dataclass
class PathConfig:
    """Path configuration for data, models, and outputs."""

    base_dir: Path = field(
        default_factory=lambda: Path("/content")
        if "google.colab" in __import__("sys").modules
        else Path(".")
    )
    data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None

    def __post_init__(self):
        """Set derived paths and create directories."""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.figures_dir is None:
            self.figures_dir = self.base_dir / "figures"
        if self.metrics_dir is None:
            self.metrics_dir = self.base_dir / "metrics"

        # Create directories
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.figures_dir,
            self.metrics_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data generation and processing configuration."""

    # Synthetic data generation
    n_users_small: tuple = (10, 21)
    n_days_small: tuple = (14, 22)
    n_users_large: tuple = (100, 301)
    n_days_large: tuple = (30, 61)

    # Schema validation
    required_columns: list = field(
        default_factory=lambda: [
            "user_pid",
            "ts_reminder",
            "tz_offset",
            "channel",
            "responded_within_2h",
        ]
    )

    # Channels and time buckets
    channels: list = field(default_factory=lambda: ["push", "SMS", "voice"])
    time_buckets: list = field(
        default_factory=lambda: ["morning", "afternoon", "evening", "other"]
    )


@dataclass
class ModelConfig:
    """Model training and evaluation configuration."""

    # Feature engineering
    window_size: int = 4  # For sequence models
    rolling_windows: list = field(default_factory=lambda: [7, 14])

    # Model parameters
    max_iter: int = 200
    class_weight: str = "balanced"
    test_size: float = 0.4
    val_split: float = 0.5

    # TinyTemporal
    use_tiny_temporal: Optional[bool] = None
    tiny_epochs: int = 5
    tiny_hidden: int = 16
    tiny_lr: float = 1e-3
    tiny_batch_size: int = 256

    def __post_init__(self):
        """Set derived values."""
        if self.use_tiny_temporal is None:
            env = EnvironmentConfig()
            self.use_tiny_temporal = env.in_colab and env.has_gpu


@dataclass
class BanditConfig:
    """Contextual bandit configuration."""

    epsilon_initial: float = 0.3
    epsilon_decay: str = "sqrt"  # "sqrt" or "linear"
    context_features: list = field(
        default_factory=lambda: [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "hours_since_prev",
            "ctr7",
            "ctr14",
            "ack_latency_sec",
            "lag_1",
            "lag_2",
        ]
    )
    arms: list = field(default_factory=lambda: ["push", "SMS", "voice"])
    outcome_model_min_samples: int = 20


@dataclass
class EvaluationConfig:
    """Evaluation and fairness configuration."""

    # Metrics
    threshold: float = 0.5
    n_bins_calibration: int = 10
    n_bins_ece: int = 10

    # Fairness groups
    fairness_groups: dict = field(
        default_factory=lambda: {
            "channel": ["push", "SMS", "voice"],
            "time_bucket": ["morning", "afternoon", "evening", "other"],
            "weekpart": ["weekday", "weekend"],
        }
    )

    # Stress testing
    drop_fraction: float = 0.2
    time_shift_hours: int = 2
    label_noise_fraction: float = 0.05


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls()

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return asdict(self)
