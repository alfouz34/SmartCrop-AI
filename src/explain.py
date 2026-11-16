from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from src.preprocessing import load_dataset, split_dataset, FEATURES

def main():
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")

    model = joblib.load(model_path)
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.20, random_state=42)

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Built-in Gini Impurity Feature Importance
    gini_importances = model.feature_importances_
    df_gini = pd.DataFrame({
        "feature": FEATURES,
        "importance": gini_importances
    }).sort_values("importance", ascending=True).reset_index(drop=True)

    df_gini.to_csv(output_dir / "feature_importance_rf.csv", index=False)
    print("--- RANDOM FOREST BUILT-IN FEATURE IMPORTANCES (GINI) ---")
    for _, row in df_gini.sort_values("importance", ascending=False).iterrows():
        pct = row["importance"] * 100
        print(f"  {row['feature']}: {row['importance']:.6f} ({pct:.2f}%)")

    # Horizontal Bar Chart for Built-in Gini Importance
    plt.figure(figsize=(9, 5))
    bars = plt.barh(df_gini["feature"], df_gini["importance"], color="#2ca02c", edgecolor="black")
    plt.title("SmartCrop AI - Random Forest Built-in Feature Importance (Gini)", fontsize=12, pad=12)
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10)
    plt.ylabel("Input Feature", fontsize=10)
    plt.xlim(0, max(df_gini["importance"]) * 1.18)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w * 100:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_rf.png", dpi=180)
    plt.close()

    # 2. Permutation Feature Importance on Held-Out Test Set
    perm_res = permutation_importance(
        model, X_test, y_test,
        scoring="f1_macro",
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    df_perm = pd.DataFrame({
        "feature": FEATURES,
        "mean_importance": perm_res.importances_mean,
        "std_importance": perm_res.importances_std
    }).sort_values("mean_importance", ascending=True).reset_index(drop=True)

    df_perm.to_csv(output_dir / "permutation_importance.csv", index=False)
    print("\n--- PERMUTATION FEATURE IMPORTANCE (TEST SET, F1-MACRO) ---")
    for _, row in df_perm.sort_values("mean_importance", ascending=False).iterrows():
        print(f"  {row['feature']}: {row['mean_importance']:.6f} +/- {row['std_importance']:.6f}")

    # Horizontal Bar Chart for Permutation Importance
    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        df_perm["feature"], df_perm["mean_importance"],
        xerr=df_perm["std_importance"],
        color="#1f77b4", edgecolor="black", capsize=4
    )
    plt.title("SmartCrop AI - Permutation Feature Importance (Held-Out Test Set, Macro F1)", fontsize=12, pad=12)
    plt.xlabel("Mean Decrease in Macro F1-Score when Permuted", fontsize=10)
    plt.ylabel("Input Feature", fontsize=10)
    max_x = max(df_perm["mean_importance"] + df_perm["std_importance"])
    plt.xlim(0, max_x * 1.18)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=180)
    plt.close()

    print(f"\nSaved CSVs and figures to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
