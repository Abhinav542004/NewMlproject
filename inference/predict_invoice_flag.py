
import joblib
import pandas as pd

from pathlib import Path

# Model paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "invoice_flagging"
    / "models"
    / "invoice_risk_model.pkl"
)

SCALER_PATH = (
    BASE_DIR.parent
    / "invoice_flagging"
    / "models"
    / "scaler.pkl"
)


# Load trained invoice classification model
def load_model(model_path=MODEL_PATH):
    """Load trained invoice risk classification model."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


# Load feature scaler
def load_scaler(scaler_path=SCALER_PATH):
    """Load trained StandardScaler."""

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    return joblib.load(scaler_path)


# Predict invoice risk
def predict_invoice_flag(input_data, model=None, scaler=None):
    """
    Predict invoice risk for new vendor invoices.

    Parameters:
        input_data: Dictionary containing invoice features.
        model: Optional pre-loaded classifier (uses load_model() if None).
        scaler: Optional pre-loaded scaler (uses load_scaler() if None).

    Returns:
        DataFrame with invoice risk predictions.
    """

    # Load model and scaler if not provided
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    # Define required features in training order
    features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars"
    ]

    # Convert input into DataFrame
    input_df = pd.DataFrame(input_data)

    # Validate required features
    missing_features = [
        col for col in features
        if col not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    input_df = input_df[features]

    # Scale input features
    scaled_data = scaler.transform(input_df)

    # Predict invoice risk
    predictions = model.predict(scaled_data)

    probabilities = model.predict_proba(scaled_data)[:, 1]

    # Add predictions to output
    result_df = input_df.copy()

    result_df["Predicted_Flag"] = predictions

    result_df["Risk_Probability"] = (
        probabilities * 100
    ).round(2)

    result_df["Risk_Status"] = result_df[
        "Predicted_Flag"
    ].map({
        0: "Not Flagged",
        1: "Potentially Risky"
    })

    return result_df


# Main execution
if __name__ == "__main__":

    # Example invoice data
    sample_data = {
        "invoice_quantity": [100, 250, 50, 500],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [250, 150, 80, 20],
        "total_item_quantity": [100, 250, 50, 500],
        "total_item_dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_invoice_flag(sample_data)

    print("\nInvoice Risk Predictions:")
    print(prediction.to_string(index=False))
