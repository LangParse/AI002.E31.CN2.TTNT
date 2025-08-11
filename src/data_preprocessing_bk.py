import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from src.utils import time_to_minutes


def load_and_preprocess(file_path: str, is_train: bool = True):
    """Load data and create engineered features."""
    df = pd.read_csv(file_path)

    # Convert time columns to minutes
    df["scheduled_time_min"] = df["scheduled_time"].apply(time_to_minutes)
    df["time_taken_min"] = df["time_taken"].apply(time_to_minutes)
    df["bed_time_min"] = df["bed_time"].apply(time_to_minutes)

    # Create time-based features
    df["time_deviation"] = df["time_taken_min"] - df["scheduled_time_min"]
    df["time_before_bed"] = df["bed_time_min"] - df["scheduled_time_min"]
    df["is_morning_med"] = (df["scheduled_time_min"] < 12 * 60).astype(int)
    df["is_evening_med"] = (df["scheduled_time_min"] >= 18 * 60).astype(int)

    # Create blood pressure features
    if (
        "blood_pressure_systolic" in df.columns
        and "blood_pressure_diastolic" in df.columns
    ):
        df["bp_systolic"] = df["blood_pressure_systolic"]
        df["bp_diastolic"] = df["blood_pressure_diastolic"]
        df["bp_pulse_pressure"] = (
            df["bp_systolic"] - df["bp_diastolic"]
        )  # Pulse pressure
        df["bp_mean_arterial"] = df["bp_diastolic"] + (
            df["bp_pulse_pressure"] / 3
        )  # Mean arterial pressure

        # Blood pressure classification
        df["bp_category"] = pd.cut(
            df["bp_systolic"],
            bins=[0, 120, 130, 140, 180, float("inf")],
            labels=[
                "normal",
                "elevated",
                "hypertension_stage_1",
                "hypertension_stage_2",
                "hypertensive_crisis",
            ],
        )

        # Flag abnormal blood pressure (>140 sys or >90 dia)
        df["bp_abnormal"] = (
            (df["bp_systolic"] > 140) | (df["bp_diastolic"] > 90)
        ).astype(int)

    # Logical imputation: fill missing time_taken for taken medications
    # if is_train or target_col not in df.columns:
    #     mask = pd.isna(df["time_taken_min"]) & (df.get(target_col, 1) == 1)
    #     df.loc[mask, "time_taken_min"] = df.loc[mask, "scheduled_time_min"]

    # Create age-based features
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 18, 22, 30, 50, 65, 80, float("inf")],
            labels=[
                "child",
                "young_adult",
                "adult",
                "middle_aged",
                "senior",
                "elderly",
                "super_elderly",
            ],
        )
        df["is_elderly"] = (df["age"] >= 65).astype(int)

    # Drop original time and BP columns after transformation
    original_time_cols = ["scheduled_time", "time_taken", "bed_time"]
    original_bp_cols = (
        ["blood_pressure_systolic", "blood_pressure_diastolic"]
        if "blood_pressure_systolic" in df.columns
        else []
    )

    drop_cols = original_time_cols + original_bp_cols
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Identify feature types
    categorical_features = [
        col
        for col in [
            "occupation",
            "income_level",
            "living_status",
            "comorbidities",
            "medication_name",
            "bp_category",
            "age_group",
        ]
        if col in df.columns
    ]

    binary_features = [
        col
        for col in ["is_morning_med", "is_evening_med", "bp_abnormal", "is_elderly"]
        if col in df.columns
    ]

    # All remaining columns are numerical
    exclude_cols = (
        categorical_features + binary_features
        # + ([target_col] if target_col in df.columns else [])
    )
    numerical_features = [col for col in df.columns if col not in exclude_cols]

    # Prepare final output
    if is_train:
        X = df.drop(columns=["taken"])
        y = df["taken"]
    else:
        X = df.drop(columns=["taken"]) if "taken" in df.columns else df
        y = df["taken"] if "taken" in df.columns else None

    return X, y, categorical_features, numerical_features, binary_features


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


def get_advanced_preprocessor(
    categorical_features, numerical_features, binary_features=None
):
    """Create preprocessing pipeline with appropriate transformers for each feature type."""
    transformers = []

    # Numerical: KNN impute + robust scaling
    if numerical_features:
        num_pipeline = Pipeline(
            [
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", RobustScaler()),  # Robust to outliers
            ]
        )
        transformers.append(("num", num_pipeline, numerical_features))

    # Categorical: fill missing + one-hot encode
    if categorical_features:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, drop="if_binary"
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    # Binary: impute with most frequent value
    if binary_features:
        binary_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent"))]
        )
        transformers.append(("binary", binary_pipeline, binary_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        sparse_threshold=0,  # Return dense array
    )
