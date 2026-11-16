import numpy as np
import pandas as pd
from src.predict import recommend_crops

class DummyModelList:
    classes_ = ["rice", "maize", "coffee"]
    def predict_proba(self, X):
        assert list(X.columns) == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        return [[0.2, 0.7, 0.1]]

class DummyModelArray:
    classes_ = np.array(["rice", "maize", "coffee"])
    def predict_proba(self, X):
        assert list(X.columns) == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        return np.array([[0.2, 0.7, 0.1]])

def test_top_three_recommendations():
    values = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 25, "humidity": 80,
        "ph": 6.5, "rainfall": 100
    }
    # Test with list-based probabilities
    result_list = recommend_crops(values, model=DummyModelList(), top_n=3)
    assert result_list[0]["crop"] == "maize"
    assert len(result_list) == 3

    # Test with numpy array-based probabilities
    result_arr = recommend_crops(values, model=DummyModelArray(), top_n=3)
    assert result_arr[0]["crop"] == "maize"
    assert len(result_arr) == 3
