import pandas as pd
from src.preprocessing import FEATURES, split_dataset

def test_split_shapes_and_stratification():
    rows = []
    for crop in ["rice", "maize"]:
        for i in range(10):
            rows.append({
                "N": i, "P": i + 1, "K": i + 2,
                "temperature": 25.0, "humidity": 70.0,
                "ph": 6.5, "rainfall": 100.0, "label": crop
            })
    df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = split_dataset(df)
    assert list(X_train.columns) == FEATURES
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert set(y_train.unique()) == {"rice", "maize"}
    assert set(y_test.unique()) == {"rice", "maize"}
