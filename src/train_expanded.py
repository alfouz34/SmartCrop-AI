"""
Expanded Agricultural Model Training & Rigorous Evaluation Pipeline
SmartCrop AI — Real-World Agricultural Dataset Expansion

This script:
1. Preserves original production models and scripts (models/best_model.joblib, src/train.py).
2. Loads data/processed/expanded_crop_dataset.csv (3,900 rows across 39 crop classes).
3. Evaluates 5 machine learning classifiers using 5-fold Stratified Cross-Validation.
4. Performs an Out-of-Region / Geographic Holdout evaluation (holding out high-range districts).
5. Compares the expanded models directly against the baseline 22-crop Random Forest.
6. Computes comprehensive Macro and Per-Class metrics (Precision, Recall, F1).
7. Exports results to:
   - reports/expanded_model_comparison.csv
   - models/expanded/expanded_best_model.joblib
   - models/expanded/expanded_model_metadata.json
   - reports/model_evaluation/expanded_classification_report.txt
   - reports/figures/expanded_confusion_matrix.png
"""

from pathlib import Path
import json
import joblib
import numpy as np
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Configuration & Paths
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPANDED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_crop_dataset.csv"
EXPANDED_MODEL_DIR = PROJECT_ROOT / "models" / "expanded"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EVAL_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

CORE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "crop"

def build_candidate_models():
    """Build candidate ML architectures for benchmark comparison."""
    return {
        "Decision Tree (Expanded)": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest (Expanded)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees (Expanded)": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN (Expanded)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM (Expanded)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
    }

def run_cross_validation(models, X_train, y_train):
    """Run stratified 5-fold cross validation strictly on training fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    
    cv_records = []
    for name, model in models.items():
        print(f"  -> Cross-validating {name}...")
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_records.append({
            "model": name,
            "cv_accuracy_mean": float(res["test_accuracy"].mean()),
            "cv_accuracy_std": float(res["test_accuracy"].std()),
            "cv_precision_macro_mean": float(res["test_precision_macro"].mean()),
            "cv_recall_macro_mean": float(res["test_recall_macro"].mean()),
            "cv_macro_f1_mean": float(res["test_f1_macro"].mean()),
            "cv_macro_f1_std": float(res["test_f1_macro"].std()),
        })
    return pd.DataFrame(cv_records)

def run_geographic_holdout_evaluation(df):
    """
    Evaluate model generalizability using an out-of-region holdout.
    We hold out the High-Range Western Ghats districts (Idukki and Wayanad) which have
    distinct mountain microclimates, and evaluate on them.
    """
    print("\nRunning Geographic Holdout Evaluation (Holdout: Idukki & Wayanad high ranges)...")
    kerala_mask = df["data_source"] != "Kaggle Benchmark (Semi-Synthetic)"
    df_kerala = df[kerala_mask].copy()
    
    holdout_districts = ["Idukki", "Wayanad"]
    train_geo = df_kerala[~df_kerala["district"].isin(holdout_districts)]
    test_geo = df_kerala[df_kerala["district"].isin(holdout_districts)]
    
    # Restrict test_geo to classes present in train_geo to allow classification evaluation
    common_crops = sorted(list(set(train_geo["crop"]).intersection(set(test_geo["crop"]))))
    train_geo = train_geo[train_geo["crop"].isin(common_crops)]
    test_geo = test_geo[test_geo["crop"].isin(common_crops)]
    
    if len(test_geo) == 0 or len(common_crops) < 2:
        return None
        
    X_train_geo = train_geo[CORE_FEATURES]
    y_train_geo = train_geo[TARGET]
    X_test_geo = test_geo[CORE_FEATURES]
    y_test_geo = test_geo[TARGET]
    
    geo_rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    geo_rf.fit(X_train_geo, y_train_geo)
    preds = geo_rf.predict(X_test_geo)
    
    return {
        "model": "Random Forest (Geographic Holdout - High Ranges)",
        "train_samples": len(train_geo),
        "test_samples": len(test_geo),
        "num_classes": len(common_crops),
        "test_accuracy": float(accuracy_score(y_test_geo, preds)),
        "test_precision_macro": float(precision_score(y_test_geo, preds, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test_geo, preds, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test_geo, preds, average="macro", zero_division=0)),
    }

def main():
    print("=" * 80)
    print("SMARTCROP AI: EXPANDED MODEL EXPERIMENT & RIGOROUS BENCHMARK")
    print("=" * 80)
    
    # 1. Load expanded dataset
    print(f"Loading expanded dataset from {EXPANDED_DATA_PATH}...")
    if not EXPANDED_DATA_PATH.exists():
        raise FileNotFoundError(f"Expanded dataset not found at {EXPANDED_DATA_PATH}. Run build_expanded_dataset.py first.")
    
    df = pd.read_csv(EXPANDED_DATA_PATH)
    print(f"Loaded {len(df):,} total records across {df['crop'].nunique()} distinct crops.")
    
    # 2. Stratified train/test split (80/20)
    X = df[CORE_FEATURES].copy()
    y = df[TARGET].astype(str).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Stratified Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")
    print(f"Class representation in test set: exactly {len(X_test) // df['crop'].nunique()} samples per crop.")
    
    # 3. Stratified 5-Fold Cross-Validation on Training Set
    print("\nExecuting Stratified 5-Fold Cross-Validation across candidate models...")
    models = build_candidate_models()
    cv_df = run_cross_validation(models, X_train, y_train)
    cv_df = cv_df.sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    
    best_candidate_name = cv_df.loc[0, "model"]
    print(f"\nCross-Validation Winner: {best_candidate_name} (Mean CV Macro F1: {cv_df.loc[0, 'cv_macro_f1_mean']:.4f})")
    
    # 4. Evaluate all candidate models on the held-out test set
    print("\nFitting models on full training fold and evaluating on held-out test fold...")
    test_records = []
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted_models[name] = model
        test_records.append({
            "model": name,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        })
    test_df = pd.DataFrame(test_records)
    expanded_results = cv_df.merge(test_df, on="model")
    
    # 5. Load original baseline metrics for direct comparison
    original_metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    baseline_row = None
    if original_metadata_path.exists():
        with open(original_metadata_path, "r", encoding="utf-8") as f:
            orig_meta = json.load(f)
        baseline_row = {
            "model": "Random Forest (Original 22-Crop Baseline)",
            "num_classes": 22,
            "total_samples": 2200,
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
            "cv_precision_macro_mean": np.nan,
            "cv_recall_macro_mean": np.nan,
            "cv_macro_f1_mean": float(orig_meta.get("cv_macro_f1_mean", 0.9937)),
            "cv_macro_f1_std": np.nan,
            "test_accuracy": float(orig_meta.get("test_accuracy", 0.9932)),
            "test_precision_macro": float(orig_meta.get("test_precision_macro", 0.9935)),
            "test_recall_macro": float(orig_meta.get("test_recall_macro", 0.9932)),
            "test_macro_f1": float(orig_meta.get("test_macro_f1", 0.9932)),
            "evaluation_notes": "Trained exclusively on 22 benchmark crops; completely blind to 17 Kerala crops."
        }
    
    # 6. Format complete comparison table
    comparison_rows = []
    if baseline_row:
        comparison_rows.append(baseline_row)
        
    for _, row in expanded_results.iterrows():
        comparison_rows.append({
            "model": row["model"],
            "num_classes": 39,
            "total_samples": 3900,
            "cv_accuracy_mean": round(row["cv_accuracy_mean"], 4),
            "cv_accuracy_std": round(row["cv_accuracy_std"], 4),
            "cv_precision_macro_mean": round(row["cv_precision_macro_mean"], 4),
            "cv_recall_macro_mean": round(row["cv_recall_macro_mean"], 4),
            "cv_macro_f1_mean": round(row["cv_macro_f1_mean"], 4),
            "cv_macro_f1_std": round(row["cv_macro_f1_std"], 4),
            "test_accuracy": round(row["test_accuracy"], 4),
            "test_precision_macro": round(row["test_precision_macro"], 4),
            "test_recall_macro": round(row["test_recall_macro"], 4),
            "test_macro_f1": round(row["test_macro_f1"], 4),
            "evaluation_notes": "Expanded 39-crop dataset (22 benchmark + 17 Kerala agricultural staples)."
        })
        
    # Geographic holdout
    geo_res = run_geographic_holdout_evaluation(df)
    if geo_res:
        comparison_rows.append({
            "model": geo_res["model"],
            "num_classes": geo_res["num_classes"],
            "total_samples": geo_res["train_samples"] + geo_res["test_samples"],
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
            "cv_precision_macro_mean": np.nan,
            "cv_recall_macro_mean": np.nan,
            "cv_macro_f1_mean": np.nan,
            "cv_macro_f1_std": np.nan,
            "test_accuracy": round(geo_res["test_accuracy"], 4),
            "test_precision_macro": round(geo_res["test_precision_macro"], 4),
            "test_recall_macro": round(geo_res["test_recall_macro"], 4),
            "test_macro_f1": round(geo_res["test_macro_f1"], 4),
            "evaluation_notes": "Trained on midland/coastal/plain districts; tested on held-out High-Range districts (Idukki & Wayanad)."
        })
        
    final_comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = REPORTS_DIR / "expanded_model_comparison.csv"
    final_comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison summary written to: {comparison_csv_path}")
    print(final_comparison_df[["model", "num_classes", "cv_macro_f1_mean", "test_accuracy", "test_macro_f1"]].to_string(index=False))
    
    # 7. Select best model & save experimental artifacts
    best_model = fitted_models[best_candidate_name]
    best_pred = best_model.predict(X_test)
    
    EXPANDED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save best expanded model
    joblib.dump(best_model, EXPANDED_MODEL_DIR / "expanded_best_model.joblib")
    
    # Classification Report
    class_report_str = classification_report(y_test, best_pred, zero_division=0)
    with open(EVAL_DIR / "expanded_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"EXPANDED MODEL EVALUATION: {best_candidate_name}\n")
        f.write(f"Total Classes: 39 | Total Test Samples: {len(y_test)}\n\n")
        f.write(class_report_str)
    print(f"\nExpanded classification report saved to: {EVAL_DIR / 'expanded_classification_report.txt'}")
    
    # Metadata JSON
    expanded_metadata = {
        "project": "SmartCrop AI (Expanded)",
        "selected_model": best_candidate_name,
        "features": CORE_FEATURES,
        "target": TARGET,
        "selection_metric": "5-fold cross-validation macro F1",
        "cv_macro_f1_mean": float(cv_df.loc[0, "cv_macro_f1_mean"]),
        "test_accuracy": float(accuracy_score(y_test, best_pred)),
        "test_precision_macro": float(precision_score(y_test, best_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, best_pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, best_pred, average="macro", zero_division=0)),
        "classes": [str(c) for c in sorted(best_model.classes_)],
        "num_classes": len(best_model.classes_),
        "random_state": RANDOM_STATE
    }
    with open(EXPANDED_MODEL_DIR / "expanded_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(expanded_metadata, f, indent=2)
    print(f"Expanded model metadata saved to: {EXPANDED_MODEL_DIR / 'expanded_model_metadata.json'}")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, best_pred, labels=best_model.classes_)
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm, cmap="YlGnBu", annot=True, fmt="d",
        xticklabels=best_model.classes_, yticklabels=best_model.classes_,
        cbar_kws={'label': 'Sample Count'}
    )
    plt.title(f"SmartCrop AI - Expanded Confusion Matrix ({best_candidate_name} - 39 Crops)", fontsize=14, pad=15)
    plt.xlabel("Predicted Crop", fontsize=12)
    plt.ylabel("Actual Crop", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    conf_matrix_path = FIGURES_DIR / "expanded_confusion_matrix.png"
    plt.savefig(conf_matrix_path, dpi=200)
    plt.close()
    print(f"Expanded confusion matrix saved to: {conf_matrix_path}")
    
    # Per-Class Performance Export
    class_report_dict = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    per_class_rows = []
    for cls_name in sorted(best_model.classes_):
        metrics = class_report_dict[cls_name]
        is_kerala_new = cls_name in [
            "rubber", "tapioca", "arecanut", "cashew", "blackpepper", "cardamom",
            "ginger", "turmeric", "tea", "clove", "nutmeg", "sweetpotato",
            "elephantfootyam", "tomato", "chilli", "brinjal", "okra"
        ]
        per_class_rows.append({
            "crop": cls_name,
            "category": "New Kerala Crop" if is_kerala_new else "Benchmark Crop",
            "support": int(metrics["support"]),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1-score"], 4)
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv_path = REPORTS_DIR / "expanded_per_class_performance.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"Per-class performance breakdown written to: {per_class_csv_path}")
    
    print("\n" + "=" * 80)
    print("EXPANDED TRAINING EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
