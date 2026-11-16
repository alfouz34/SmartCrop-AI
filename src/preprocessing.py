from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = Path("data/raw/Crop_recommendation.csv")
if not DEFAULT_DATA_PATH.exists() and (BASE_DIR / "data" / "raw" / "Crop_recommendation.csv").exists():
    DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "Crop_recommendation.csv"


def load_dataset(path=DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Crop_recommendation.csv in data/raw/."
        )
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[required].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if df.duplicated(subset=required).any():
        print("Warning: duplicate rows detected. They are retained for transparent evaluation.")
    return df


def split_dataset(df, test_size=0.20, random_state=42):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(str).copy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
