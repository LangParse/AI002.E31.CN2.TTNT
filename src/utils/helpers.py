"""
Helper utilities for AI Medication Reminder.

Contains common utility functions for logging, file operations, and data handling.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def setup_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("ai_medication_reminder")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_results(data: Any, path: Path, format: str = "json") -> None:
    """
    Save results to file in specified format.

    Args:
        data: Data to save
        path: Path to save file
        format: Format to save in ("json", "pickle", "csv")
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif format.lower() == "pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif format.lower() == "csv" and isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Results saved to {path}")


def load_results(path: Path, format: str = "json") -> Any:
    """
    Load results from file.

    Args:
        path: Path to load from
        format: Format to load ("json", "pickle", "csv")

    Returns:
        Loaded data
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format.lower() == "json":
        with open(path, "r") as f:
            return json.load(f)
    elif format.lower() == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif format.lower() == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["env", "paths", "data", "model", "bandit", "evaluation"]

    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False

    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Width of the header
    """
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)
