from datetime import datetime
import numpy as np
import pandas as pd


def time_to_minutes(time_str: str) -> float:
    """
    Convert time string in HH:MM format to minutes since 00:00.

    Args:
        time_str: Time string in "HH:MM" format, "None", or NaN

    Returns:
        float: Number of minutes since 00:00, or np.nan if invalid

    Examples:
        >>> time_to_minutes("09:30")
        570.0
        >>> time_to_minutes("00:00")
        0.0
        >>> time_to_minutes("23:59")
        1439.0
    """

    # Handle None and NaN values
    if time_str in ["None", ""] or pd.isna(time_str):
        return np.nan

    # Handle empty or whitespace strings
    if not isinstance(time_str, str) or not time_str.strip():
        return np.nan

    try:
        dt = datetime.strptime(time_str, "%H:%M")
        return float(dt.hour * 60 + dt.minute)
    except ValueError:
        # Specifially catch parsing errors
        return np.nan
    except TypeError:
        # Handle cases where time_str is not string-like
        return np.nan
