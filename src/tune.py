from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.preprocessing import load_dataset, split_dataset

def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        base,
        param_distributions=params,
        n_iter=20,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, "models/tuned_random_forest.joblib")
    result = {
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    pd.Series(result, dtype=object).to_json(
        "models/tuned_random_forest_results.json", indent=2
    )
    print(result)
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()
