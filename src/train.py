from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.preprocessing import load_dataset, split_dataset, FEATURES

RANDOM_STATE = 42

def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
    }

def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    cv_rows = []
    for name, model in build_models().items():
        print(f"Stratified 5-fold cross-validating {name} on training data...")
        cv_res = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_rows.append({
            "model": name,
            "cv_accuracy_mean": float(cv_res["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv_res["test_accuracy"].std()),
            "cv_precision_macro_mean": float(cv_res["test_precision_macro"].mean()),
            "cv_recall_macro_mean": float(cv_res["test_recall_macro"].mean()),
            "cv_macro_f1_mean": float(cv_res["test_f1_macro"].mean()),
            "cv_macro_f1_std": float(cv_res["test_f1_macro"].std()),
        })

    cv_df = pd.DataFrame(cv_rows).sort_values(
        "cv_macro_f1_mean", ascending=False
    ).reset_index(drop=True)

    best_name = cv_df.loc[0, "model"]
    print(f"\nModel selection completed using 5-fold CV macro F1 on training data.")
    print(f"Selected Best Model: {best_name} (Mean CV Macro F1: {cv_df.loc[0, 'cv_macro_f1_mean']:.4f})")

    # Evaluate models on held-out test set only after model selection
    test_rows = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        test_rows.append({
            "model": name,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        })

    test_df = pd.DataFrame(test_rows)
    results = cv_df.merge(test_df, on="model")

    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    Path("models").mkdir(exist_ok=True)
    Path("reports/model_evaluation").mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, "models/best_model.joblib")
    results.to_csv("models/model_comparison.csv", index=False)

    metadata = {
        "project": "SmartCrop AI",
        "selected_model": best_name,
        "features": FEATURES,
        "target": "label",
        "selection_metric": "5-fold cross-validation macro F1",
        "cv_macro_f1_mean": float(cv_df.loc[0, "cv_macro_f1_mean"]),
        "test_accuracy": float(accuracy_score(y_test, best_pred)),
        "test_precision_macro": float(precision_score(y_test, best_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, best_pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, best_pred, average="macro")),
        "classes": [str(c) for c in best_model.classes_],
        "random_state": RANDOM_STATE,
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report = classification_report(y_test, best_pred, zero_division=0)
    with open("reports/model_evaluation/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected model: {best_name}\n\n{report}")

    cm = confusion_matrix(y_test, best_pred, labels=best_model.classes_)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, cmap="Greens", annot=True, fmt="d",
        xticklabels=best_model.classes_, yticklabels=best_model.classes_
    )
    plt.title(f"SmartCrop AI - Confusion Matrix ({best_name})")
    plt.xlabel("Predicted crop")
    plt.ylabel("Actual crop")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/model_evaluation/confusion_matrix.png", dpi=180)
    plt.close()

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nClassification report:\n", report)
    print("\nSaved model and evaluation artifacts.")

if __name__ == "__main__":
    main()
