import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import joblib


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    print("Model Evaluation Metrics:", metrics)
    return metrics, y_pred


def visualize_results(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Cofusion Matrix for - {model_name}")
    # plt.savefig(f"reports/confusion_matrix_{model_name}.png")

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    # plt.savefig(f"reports/roc_curve_{model_name}.png")

    # Feature Importance (nếu là tree-based)
    if hasattr(model.named_steps["model"], "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
        features = model.named_steps["preprocessor"].get_feature_names_out()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title(f"Feature Importances - {model_name}")
        # plt.savefig(f"reports/feature_importances_{model_name}.png")


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
