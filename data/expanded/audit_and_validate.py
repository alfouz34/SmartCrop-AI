"""
Phase 5 Deep Audit & Scientific Validation Script
SmartCrop AI
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPANDED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_crop_dataset.csv"
CORE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RANDOM_STATE = 42

def run_audit():
    df = pd.read_csv(EXPANDED_DATA_PATH)
    print("=" * 80)
    print(f"EXPANDED DATASET AUDIT (Total Rows: {len(df)})")
    print("=" * 80)
    
    # 1. Exact Duplicates
    exact_all = df.duplicated().sum()
    exact_features = df.duplicated(subset=CORE_FEATURES).sum()
    print(f"Exact duplicates across all 14 columns: {exact_all}")
    print(f"Exact duplicates across 7 core features: {exact_features}")
    
    # 2. Near-Duplicates (using Nearest Neighbors on standardized core features)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CORE_FEATURES])
    
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    # distances[:, 1] is distance to the nearest neighbor other than self
    min_dist = distances[:, 1]
    near_dups_01 = (min_dist < 0.05).sum()
    near_dups_02 = (min_dist < 0.10).sum()
    print(f"Near-duplicates with normalized Euclidean distance < 0.05: {near_dups_01}")
    print(f"Near-duplicates with normalized Euclidean distance < 0.10: {near_dups_02}")
    print(f"Minimum pairwise normalized distance: {min_dist.min():.5f}")
    
    # 3. Provenance Classification
    # - Original 2,200 records: Augmented/compiled benchmark with independently unverified underlying observation provenance.
    # - Additional 1,700 Kerala records: Synthetic/simulated observations generated from agronomic and regional parameter ranges.
    # - Combined dataset: Mixed-provenance research dataset, not verified farm-level ground truth.
    
    bench_count = (df["data_source"] == "Kaggle Benchmark (Semi-Synthetic)").sum()
    kerala_count = len(df) - bench_count

    print("\nPROVENANCE CATEGORY BREAKDOWN:")
    print("A. Genuine linked farm-level observation: 0 (0.0%)")
    print("B. Genuine experimental or field-trial observation: 0 (0.0%)")
    print("C. Curated observation assembled from multiple published sources: 0 (0.0%)")
    print(f"D. Synthetic/simulated observation from agronomic & regional ranges: {kerala_count} ({kerala_count/len(df)*100:.2f}%) [Kerala cohort]")
    print(f"E. Augmented/compiled benchmark with unverified provenance: {bench_count} ({bench_count/len(df)*100:.2f}%) [Original Kaggle cohort]")
    print(f"-> Combined dataset: Mixed-provenance research dataset ({len(df)} records), not verified farm-level ground truth.")
    
    # 4. Detailed Geographic Holdout Inspection
    print("\n" + "=" * 80)
    print("GEOGRAPHIC HOLDOUT DETAILED AUDIT")
    print("=" * 80)
    kerala_mask = df["data_source"] != "Kaggle Benchmark (Semi-Synthetic)"
    df_kerala = df[kerala_mask].copy()
    
    holdout_districts = ["Idukki", "Wayanad"]
    train_geo = df_kerala[~df_kerala["district"].isin(holdout_districts)].copy()
    test_geo = df_kerala[df_kerala["district"].isin(holdout_districts)].copy()
    
    print(f"Total Kerala Records: {len(df_kerala)}")
    print(f"Records from Midland/Coastal/Plains (Non-Holdout): {len(train_geo)}")
    print(f"Records from High-Ranges (Idukki & Wayanad): {len(test_geo)}")
    
    print("\nCrops present in Idukki/Wayanad (Holdout Districts):")
    print(test_geo["crop"].value_counts())
    
    print("\nCrops present in Non-Holdout Districts:")
    print(train_geo["crop"].value_counts())
    
    common_crops = sorted(list(set(train_geo["crop"]).intersection(set(test_geo["crop"]))))
    exclusive_holdout = sorted(list(set(test_geo["crop"]) - set(train_geo["crop"])))
    exclusive_train = sorted(list(set(train_geo["crop"]) - set(test_geo["crop"])))
    print(f"\nCrops common to both regions ({len(common_crops)}): {common_crops}")
    print(f"Crops EXCLUSIVE to high ranges ({len(exclusive_holdout)}): {exclusive_holdout}")
    print(f"Crops EXCLUSIVE to plains/midlands ({len(exclusive_train)}): {exclusive_train}")
    
    # Notice: Cardamom and Tea are strictly high-range crops in Kerala!
    # They ONLY appear in Idukki and Wayanad in df_kerala!
    # So if you train only on non-holdout districts, the model has NEVER SEEN cardamom or tea!
    # That is why in train_expanded.py, common_crops had 5 crops!
    
    train_geo_common = train_geo[train_geo["crop"].isin(common_crops)]
    test_geo_common = test_geo[test_geo["crop"].isin(common_crops)]
    
    print(f"\nFiltered Common Sample Sizes: Train = {len(train_geo_common)}, Test = {len(test_geo_common)}")
    print("Common crops test set distribution:")
    print(test_geo_common["crop"].value_counts())
    
    X_train_geo = train_geo_common[CORE_FEATURES]
    y_train_geo = train_geo_common["crop"]
    X_test_geo = test_geo_common[CORE_FEATURES]
    y_test_geo = test_geo_common["crop"]
    
    rf_geo = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf_geo.fit(X_train_geo, y_train_geo)
    geo_preds = rf_geo.predict(X_test_geo)
    
    acc = accuracy_score(y_test_geo, geo_preds)
    p_macro = precision_score(y_test_geo, geo_preds, average="macro", zero_division=0)
    r_macro = recall_score(y_test_geo, geo_preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_geo, geo_preds, average="macro", zero_division=0)
    
    print(f"\nGeographic Holdout Reproducibility Check:")
    print(f"Test Accuracy: {acc:.4f} (Expected: 0.7419)")
    print(f"Macro Precision: {p_macro:.4f} (Expected: 0.7628)")
    print(f"Macro Recall: {r_macro:.4f} (Expected: 0.7786)")
    print(f"Macro F1: {f1_macro:.4f} (Expected: 0.7232)")
    print("\nPer-class report on Geographic Holdout:")
    print(classification_report(y_test_geo, geo_preds, zero_division=0))
    
    # 5. Grouped Cross-Validation on Kerala Subset
    print("\n" + "=" * 80)
    print("DISTRICT-GROUPED CROSS-VALIDATION (LEAVE-DISTRICT-OUT)")
    print("=" * 80)
    # If we use GroupKFold on district, can a model trained on some districts predict crops in unseen districts?
    gkf = GroupKFold(n_splits=5)
    groups = df_kerala["district"]
    X_k = df_kerala[CORE_FEATURES]
    y_k = df_kerala["crop"]
    
    g_accs, g_f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_k, y_k, groups)):
        X_tr, y_tr = X_k.iloc[train_idx], y_k.iloc[train_idx]
        X_va, y_va = X_k.iloc[val_idx], y_k.iloc[val_idx]
        
        # Check which classes are in validation but not train
        val_classes_in_train = y_va.isin(y_tr.unique())
        
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        
        # We can only predict classes that were in train!
        # If an exclusive class like tea or cardamom is in val fold only, it will get 0% accuracy
        preds = rf.predict(X_va)
        fold_acc = accuracy_score(y_va, preds)
        fold_f1 = f1_score(y_va, preds, average="macro", zero_division=0)
        g_accs.append(fold_acc)
        g_f1s.append(fold_f1)
        val_dists = groups.iloc[val_idx].unique()
        print(f"Fold {fold+1} (Districts: {list(val_dists)}): Accuracy = {fold_acc:.4f}, Macro F1 = {fold_f1:.4f}")
        
    print(f"Mean District-Grouped CV Accuracy: {np.mean(g_accs):.4f} +/- {np.std(g_accs):.4f}")
    print(f"Mean District-Grouped CV Macro F1: {np.mean(g_f1s):.4f} +/- {np.std(g_f1s):.4f}")

if __name__ == "__main__":
    run_audit()
