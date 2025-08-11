from src.data_preprocessing import (
    get_preprocessor,
    load_and_preprocess,
)
from src.model_training import load_config, train_model
from src.model_evaluation import evaluate_model, visualize_results, save_model

if __name__ == "__main__":
    # Load configuration
    config = load_config("config/config.yaml")

    # Load and preprocess data
    X_train, y_train, categorical_features, numerical_features = load_and_preprocess(
        "data/synthetic_hypertension_train.csv", is_train=True
    )
    X_test, y_test, _, _ = load_and_preprocess(
        "data/synthetic_hypertension_train.csv", is_train=False
    )

    # Get preprocessor
    preprocessor = get_preprocessor(categorical_features, numerical_features)

    # Train model
    models = {}
    for model_name in ["random_forest", "xgboost"]:
        model = train_model(model_name, X_train, y_train, preprocessor, config)
        models[model_name] = model

    # Evaluate and visualize results
    best_model = None
    best_f1 = 0
    for model_name, model in models.items():
        metrics, _ = evaluate_model(model, X_test, y_test)
        visualize_results(model, X_test, y_test, model_name)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model

    # Save model to deploy
    save_model(best_model, "models/hypertension_model_v1.pkl")

    print("Model training and evaluation completed successfully.")
