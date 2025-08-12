import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    save_path = f"./visualizations/{model_name}_cm.png"
    os.makedirs(
        os.path.dirname(save_path), exist_ok=True
    )  # Create directory if not exists
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_feature_importance(pipeline, feature_names, model_name):
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        importances = pipeline.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f"Feature Importance - {model_name}")
        save_path = f"./visualizations/{model_name}_fi.png"
        os.makedirs(
            os.path.dirname(save_path), exist_ok=True
        )  # Create directory if not exists
        plt.savefig(save_path)
        plt.close()
        print(f"Saved feature importance to {save_path}")
