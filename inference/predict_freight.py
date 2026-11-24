
import joblib
import pandas as pd

from pathlib import Path

# Model path
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "Freight cost prediction"
    / "models"
    / "predict_freight_model.pkl"
)

# Load trained freight prediction model
def load_model(model_path=MODEL_PATH):
    """Load trained freight cost prediction model."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Freight model not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model


# Predict freight cost
def predict_freight_cost(input_data, model=None):
    """
    Predict freight cost for new vendor invoices.

    Parameters:
        input_data: Dictionary containing invoice features.
        model: Optional pre-loaded model (uses load_model() if None).

    Returns:
        DataFrame with predicted freight cost.
    """

    # Use pre-loaded model or load fresh
    if model is None:
        model = load_model()

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame(input_data)

    # Predict freight cost
    input_df["Predicted_Freight"] = model.predict(
        input_df
    ).round(2)

    return input_df


# Main execution
if __name__ == "__main__":

    # Example inference run
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)

    print("\nFreight Cost Predictions:")
    print(prediction)
