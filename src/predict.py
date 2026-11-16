from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path("models/best_model.joblib")
if not MODEL_PATH.exists() and (BASE_DIR / "models" / "best_model.joblib").exists():
    MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
_cached_model = None

def load_model(reload: bool = False):
    global _cached_model
    if _cached_model is None or reload:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found at models/best_model.joblib. Run `python -m src.train` first."
            )
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model

def recommend_crops(values, model=None, top_n=3):
    """
    Generate top-N crop recommendations based on soil nutrients and climate inputs.

    Note:
        When using models with predict_proba(), the returned scores represent
        model-estimated class probabilities (computed by averaging the class probability
        estimates produced by the individual decision trees) that have not been
        probability-calibrated.
    """
    model = model or load_model()
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(row)[0])
        classes = np.asarray(model.classes_)
        order = probabilities.argsort()[::-1][:top_n]
        return [
            {"crop": str(classes[i]), "score": float(round(probabilities[i], 4))}
            for i in order
        ]

    prediction = model.predict(row)[0]
    return [{"crop": str(prediction), "score": None}]
