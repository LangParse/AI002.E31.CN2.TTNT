import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def train_model(model_name, X_train, y_train, preprocessor):
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=42)
        param_grid = {"model__C": [0.1, 1, 10]}
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
        }
    elif model_name == "xgboost":
        model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
        param_grid = {
            "model__learning_rate": [0.01, 0.1, 0.3],
            "model__max_depth": [3, 5, 7],
            "model__n_estimators": [50, 100, 200],
        }
    elif model_name == "lightgbm":
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_grid = {
            "model__learning_rate": [0.01, 0.1, 0.3],
            "model__max_depth": [3, 5, 7],
            "model__n_estimators": [50, 100, 200],
        }
    elif model_name == "catboost":
        model = cb.CatBoostClassifier(random_state=42, verbose=0)
        param_grid = {
            "model__learning_rate": [0.01, 0.1, 0.3],
            "model__depth": [3, 5, 7],
            "model__iterations": [50, 100, 200],
        }
    else:
        raise ValueError("Model không hỗ trợ")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print(f"Best params for {model_name}: {grid_search.best_params_}")

    return best_pipeline
