"""Prediction Script for Car Price Prediction
Provides utilities to load the trained model and make predictions
for single instances and batches with input validation.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Union[str, Path] = None):
    """
    Load the trained model from a pickle file.

    Args:
        model_path (str | Path, optional): Path to the model file.
            Defaults to ../models/car_price_model.pkl relative to this script.

    Returns:
        Trained model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other loading errors.
    """
    try:
        if model_path is None:
            script_dir = Path(__file__).parent
            model_path = script_dir.parent / "models" / "car_price_model.pkl"
        else:
            model_path = Path(model_path)

        logger.info(f"Loading model from {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully.")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def _validate_input_features(
    input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Validate and convert input data to a pandas DataFrame.

    Args:
        input_data: Single dict of features, list of dicts, or DataFrame.

    Returns:
        pd.DataFrame: Validated feature dataframe.

    Raises:
        ValueError: If input format is invalid or empty.
    """
    logger.info("Validating input features...")

    # Convert to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        if not input_data:
            raise ValueError("Input list is empty.")
        if not all(isinstance(item, dict) for item in input_data):
            raise ValueError("All items in the list must be dictionaries.")
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError(
            "Invalid input type. Expected dict, list of dicts, or pandas DataFrame."
        )

    if df.empty:
        raise ValueError("Input data is empty after conversion to DataFrame.")

    # Basic cleaning: ensure no completely empty rows
    df.dropna(how="all", inplace=True)
    if df.empty:
        raise ValueError("All input rows are empty after dropping NaN rows.")

    logger.info(f"Input features validated. Shape: {df.shape}")
    return df


def predict_price(
    model,
    input_features: Union[Dict[str, Any], pd.DataFrame],
) -> float:
    """
    Predict car price for a single instance.

    Args:
        model: Trained model.
        input_features (dict | pd.DataFrame): Features for a single car.

    Returns:
        float: Predicted price.

    Raises:
        ValueError: If prediction fails or input is invalid.
    """
    logger.info("Making single prediction...")

    # Validate and convert to DataFrame
    df = _validate_input_features(input_features)

    if len(df) != 1:
        raise ValueError(
            f"Expected a single instance for prediction, got {len(df)} rows."
        )

    try:
        prediction = model.predict(df)[0]
        logger.info(f"Predicted price: {prediction:.2f}")
        return float(prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def predict_batch(
    model,
    input_features_list: Union[List[Dict[str, Any]], pd.DataFrame],
) -> np.ndarray:
    """
    Perform batch predictions for multiple cars.

    Args:
        model: Trained model.
        input_features_list (list[dict] | pd.DataFrame): Features for multiple cars.

    Returns:
        np.ndarray: Array of predicted prices.

    Raises:
        ValueError: If prediction fails or input is invalid.
    """
    logger.info("Making batch predictions...")

    # Validate and convert to DataFrame
    df = _validate_input_features(input_features_list)

    try:
        predictions = model.predict(df)
        logger.info(f"Generated {len(predictions)} predictions.")
        return np.array(predictions, dtype=float)
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise


def main():
    """
    Example usage for manual testing.
    """
    try:
        # Load model
        model = load_model()

        # Example single input (you must match the feature names used during training)
        example_input = {
            # Fill these with appropriate values based on your cleaned_data columns.
            # Example (these keys MUST match trained feature columns):
            # "symboling": 3,
            # "enginesize": 130,
            # "horsepower": 111,
            # "citympg": 21,
            # "highwaympg": 27,
            # "brand_alfa-romero": 1,
            # ...
        }

        if example_input:
            price = predict_price(model, example_input)
            print(f"Predicted price for example input: {price:.2f}")
        else:
            logger.info(
                "No example_input defined. Update 'example_input' in main() to test."
            )

    except Exception as e:
        logger.error(f"Prediction script failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()


