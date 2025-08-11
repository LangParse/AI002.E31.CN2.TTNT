import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import time_to_minutes


def load_and_preprocess(file_path, is_train=True):
    df = pd.read_csv(file_path)

    # Feature engineering: Chuyển time thành minutes
    df["scheduled_time_min"] = df["scheduled_time"].apply(time_to_minutes)
    df["time_taken_min"] = df["time_taken"].apply(time_to_minutes)
    df["bed_time_min"] = df["bed_time"].apply(time_to_minutes)

    # Điền NaN logic (dựa trên taken)
    df["time_taken_min"] = df.apply(
        lambda row: row["scheduled_time_min"]
        if pd.isna(row["time_taken_min"]) and row.get("taken", 1) == 1
        else row["time_taken_min"],
        axis=1,
    )
    df["time_taken_min"].fillna(0, inplace=True)

    # Drop columns gây data leakage hoặc không cần
    drop_cols = [
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "scheduled_time",
        "time_taken",
        "bed_time",
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Phân loại features
    categorical_features = [
        "occupation",
        "income_level",
        "living_status",
        "comorbidities",
        "medication_name",
    ]
    numerical_features = [
        col for col in df.columns if col not in categorical_features + ["taken"]
    ]

    if is_train:
        X = df.drop(columns=["taken"])
        y = df["taken"]
    else:
        X = df.drop(columns=["taken"]) if "taken" in df.columns else df
        y = df["taken"] if "taken" in df.columns else None

    return X, y, categorical_features, numerical_features


def get_preprocessor(categorical_features, numerical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
