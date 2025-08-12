from src.data_preprocessing import load_and_preprocess, get_advanced_preprocessor
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.visualization import plot_confusion_matrix, plot_feature_importance
from src.deployment import save_model
import pandas as pd

MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
TRAIN_FILE = "data/synthetic_hypertension_train.csv"  # Copy data files to data/
TEST_FILE = "data/synthetic_hypertension_test.csv"

# Load data
X_train, y_train, cat_feats, num_feats, bin_feats = load_and_preprocess(
    TRAIN_FILE, is_train=True
)
X_test, y_test, _, _, _ = load_and_preprocess(TEST_FILE, is_train=False)

preprocessor = get_advanced_preprocessor(cat_feats, num_feats, bin_feats)

results = {}
best_model = None
best_f1 = 0
best_name = ""

for model_name in MODELS:
    print(f"\nTraining {model_name}...")
    pipeline = train_model(model_name, X_train, y_train, preprocessor)

    print(f"Evaluating {model_name}...")
    metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
    results[model_name] = metrics

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model = pipeline
        best_name = model_name

    # Visualization
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Get feature names after preprocessing from the fitted pipeline
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    plot_feature_importance(pipeline, feature_names, model_name)

print("\n=== Kết quả so sánh ===")
print(pd.DataFrame(results).T)

# Save best model
save_model(best_model, f"./models/{best_name}_model.pkl")

print("Done! To deploy, run: python src/deployment.py")
