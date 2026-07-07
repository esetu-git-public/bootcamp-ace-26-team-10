# utils/prediction.py
# ─────────────────────────────────────────────────────────────────
# Handles model loading and single/batch prediction logic.
# ─────────────────────────────────────────────────────────────────
print("prediction.py is loading...")
import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "kidney_model.pkl")


def load_model_bundle(model_path: str = MODEL_PATH) -> dict:
    """
    Load the persisted model bundle from disk.

    The bundle is a dict containing:
        'model'     – trained sklearn estimator
        'encoders'  – dict of fitted encoders
        'scaler'    – fitted StandardScaler
        'classes'   – list of class label strings
        'features'  – ordered list of feature column names
        'metrics'   – dict of evaluation metrics on the test set

    Parameters
    ----------
    model_path : str  Path to the .pkl bundle.

    Returns
    -------
    dict
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run the training notebook (03_Model_Training.ipynb) first."
        )
    bundle = joblib.load(model_path)
    return bundle


def predict_single(input_df: pd.DataFrame, bundle: dict) -> tuple[str, np.ndarray]:
    """
    Predict the CKD stage for a single preprocessed input row.

    Parameters
    ----------
    input_df : pd.DataFrame  (1 × n_features, already scaled)
    bundle   : dict          Model bundle from load_model_bundle().

    Returns
    -------
    (predicted_class_label: str, probabilities: np.ndarray or None)
    """
    model   = bundle["model"]
    classes = bundle["classes"]

    pred_idx = model.predict(input_df)[0]
    label    = classes[pred_idx]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]

    return label, proba


def predict_batch(X: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch prediction on a preprocessed feature matrix.

    Parameters
    ----------
    X      : pd.DataFrame  (n_samples × n_features, already scaled)
    bundle : dict          Model bundle.

    Returns
    -------
    (predictions_array, probabilities_array or None)
    """
    model = bundle["model"]
    preds = model.predict(X)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

    return preds, proba


def get_class_label(idx: int, bundle: dict) -> str:
    """Return the human-readable class label for a numeric class index."""
    return bundle["classes"][idx]

print("prediction.py loaded successfully")