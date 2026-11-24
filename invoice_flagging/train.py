
from pathlib import Path
import joblib

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features,
    DEFAULT_DB_PATH
)

from modelling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


# Define input features
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # Create model directory
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load invoice data
    db_path = DEFAULT_DB_PATH
    print("Database path:", db_path)
    print("Database exists:", db_path.exists())

    df = load_invoice_data(db_path)

    print("Data loaded successfully!")
    print("Dataset shape:", df.shape)

    # Apply invoice risk labels
    df = apply_labels(df)

    print("\nInvoice Risk Distribution:")
    print(df[TARGET].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        model_dir / "scaler.pkl"
    )

    # Train Random Forest
    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    best_model = grid_search.best_estimator_

    # Evaluate classifier
    results = evaluate_classifier(
        best_model,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    model_path = model_dir / "invoice_risk_model.pkl"

    joblib.dump(best_model, model_path)

    print("\nModel saved successfully!")
    print("Model path:", model_path)

    print("\nFinal Evaluation Results:")
    print(results)


if __name__ == "__main__":
    main()
