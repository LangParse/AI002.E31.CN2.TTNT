import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(model_name, X_train, y_train, preprocessor, config):
    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = config["models"]["random_forest"]["param_grid"]
    elif model_name == "xgboost":
        model = xgb.XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        param_grid = config["models"]["xgboost"]["param_grid"]
    else:
        raise ValueError("Model không hỗ trợ cho tuning")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Tối ưu hóa với GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=config["cv_folds"],
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_
