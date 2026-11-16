# ✅ SmartCrop AI: Final Demonstration & Verification Checklist
## Phase 7 Actionable Evaluation, Execution & Defense Guide

**Project:** SmartCrop AI  
**Reference Document:** `reports/final_demo_checklist.md`  
**Date:** November 2026 (Updated Phase 7 Submission Freeze)  
**Audience:** System Evaluators, Academic Examiners & Project Reviewers  

---

## 1. System Preservation & Integrity Checklist

Before demonstrating the system, verify that all core production artifacts remain untouched and isolated:

```text
========================================================================================
PRODUCTION ARTIFACT PRESERVATION VERIFICATION
========================================================================================
[X] models/best_model.joblib          -> UNMODIFIED (11.03 MB, 22-class baseline intact)
[X] data/raw/Crop_recommendation.csv  -> UNMODIFIED (2,200 rows, 22 crops intact)
[X] backend/main.py                   -> UNMODIFIED (FastAPI serving production model)
[X] frontend/app.py                   -> UNMODIFIED (Streamlit UI pointing to localhost:8000)
[X] models/expanded/                  -> ISOLATED (Houses expanded_best_model.joblib)
[X] data/processed/                   -> ISOLATED (Houses expanded_crop_dataset.csv)
========================================================================================
```

---

## 2. Command Execution Reference

To reproduce any experiment or run the live application stack, execute the following commands using the project's Anaconda Python environment (`C:\Users\alfouz\anaconda3\python.exe`):

### A. Run Automated Test Suite (13 Tests)
Verifies API endpoints, prediction logic, preprocessing, and expanded schema backwards compatibility:
```powershell
& "C:\Users\alfouz\anaconda3\python.exe" -m pytest SmartCrop_AI
# Expected output: 13 passed in ~4.0s
```

### B. Launch FastAPI Backend
Starts the REST API server at `http://127.0.0.1:8000` with interactive Swagger docs at `/docs`:
```powershell
& "C:\Users\alfouz\anaconda3\python.exe" -m uvicorn backend.main:app --app-dir SmartCrop_AI --reload
```

### C. Launch Streamlit Interactive Dashboard
Starts the interactive browser interface at `http://localhost:8501`:
```powershell
& "C:\Users\alfouz\anaconda3\python.exe" -m streamlit run SmartCrop_AI/frontend/app.py
```

### D. Re-run Provenance & Forensic Audit Script
Executes nearest-neighbor duplicate detection, provenance categorization, geographic holdout reproduction, and reconciled district-grouped cross-validation:
```powershell
& "C:\Users\alfouz\anaconda3\python.exe" SmartCrop_AI/data/expanded/audit_and_validate.py
```

### E. Re-run Expanded Training Pipeline
Re-trains candidate models on the 39-crop dataset and regenerates comparison CSVs and confusion matrices:
```powershell
& "C:\Users\alfouz\anaconda3\python.exe" SmartCrop_AI/src/train_expanded.py
```

---

## 3. Pre-Flight Demonstration Inputs & Presets

When presenting the live Streamlit dashboard, demonstrate these verified baseline input presets:

| Demonstration Preset | Soil Nutrients ($N, P, K$ in kg/ha) | Climate & Chemistry ($T, H, \text{pH}, R$) | Expected Top-1 Recommendation | Model Score |
| :--- | :--- | :--- | :--- | :---: |
| **Rice (Wetland / High Rainfall)** | $N=80.0, P=48.0, K=40.0$ | $T=24.0^\circ\text{C}, H=82.0\%, \text{pH}=6.4, R=235\text{ mm}$ | **Rice** | $>0.85$ |
| **Apple (High P & K Orchard)** | $N=20.0, P=134.0, K=200.0$ | $T=22.0^\circ\text{C}, H=92.0\%, \text{pH}=6.0, R=112\text{ mm}$ | **Apple** | $>0.95$ |
| **Chickpea (Arid / Low Humidity)** | $N=40.0, P=68.0, K=80.0$ | $T=19.0^\circ\text{C}, H=17.0\%, \text{pH}=7.3, R=80\text{ mm}$ | **Chickpea** | $>0.90$ |
| **Kerala Tropical Wetland (Custom)** | $N=22.0, P=17.0, K=30.0$ | $T=27.0^\circ\text{C}, H=93.0\%, \text{pH}=6.0, R=175\text{ mm}$ | **Coconut** | $>0.80$ |

---

## 4. Evaluator & Examiner Q&A Defense Sheet

Anticipated technical questions and scientifically verified answers:

### Q1: Why did overall model accuracy drop from 99.3% on the baseline to 92.0% on the expanded model?
* **Defense Answer:** The 99.3% accuracy on the baseline dataset is an artifact of augmented, non-overlapping clusters where crops never compete for the same ecological niche. In real agriculture, multiple regional crops naturally share overlapping agronomic envelopes (e.g. ginger and turmeric in loamy soils, or brinjal and chilli in vegetable tracts). A drop to 92.0% reflects biologically realistic decision boundaries.

### Q2: What is the exact provenance of the dataset?
* **Defense Answer:** The dataset is a **mixed-provenance research dataset**, not verified farm-level ground truth:
  - The original 2,200 records represent an **augmented/compiled benchmark** with independently unverified underlying observation provenance.
  - The additional 1,700 Kerala records are **synthetic/simulated observations** generated from published agronomic and regional parameter ranges (KAU Package of Practices, Soil Health Card district fertility baselines, IMD climatology, and EARAS yield averages).

### Q3: What explains the slight variance in District-Grouped Cross-Validation metrics?
* **Defense Answer:** The variance is due to ensemble tree count:
  - $n\_estimators=100$ (Phase 5 rapid screening): Accuracy = $0.7884 \pm 0.0496$, Macro F1 = $0.6317 \pm 0.0444$.
  - $n\_estimators=300$ (Phase 6/7 authoritative benchmark matching candidate models): Accuracy = $0.7984 \pm 0.0424$, Macro F1 = $0.6319 \pm 0.0466$.
  Both are fully reproducible using the exact saved dataset, random state 42, and GroupKFold across the 14 districts.

### Q4: What does the 74.19% Geographic Holdout score mean?
* **Defense Answer:** It evaluates domain shift between two distinct agro-climatic zones. Models trained on midland and coastal plains districts were tested on unseen high-range mountain districts (Idukki and Wayanad) across 5 cross-regional crops ($N=500$). The accuracy drop from 92% to 74% demonstrates that temperature lapse rates, elevation, and mountain microclimates degrade plain-trained models.

### Q5: Why is the model classified as "NOT YET READY FOR REAL FARM RECOMMENDATIONS"?
* **Defense Answer:** Because Kerala farming is dominated by long-gestation perennial plantation crops (Rubber: 7 years to tapping; Arecanut: 6 years to harvest). A false positive recommendation based on unverified simulated data could cause catastrophic multi-year bankruptcy for a smallholder farmer. Furthermore, the 7-feature schema lacks critical covariates: topographic elevation (to gate tea/cardamom) and soil drainage class (to prevent fatal *Phytophthora* foot rot).

---

## 5. Demonstration Sign-Off Checklist

```text
========================================================================================
FINAL AUDIT SIGN-OFF
========================================================================================
[X] 1. Baseline dataset and production model preserved intact.
[X] 2. 14-column expanded dataset verified and validated (3,900 rows, 39 crops).
[X] 3. Comprehensive provenance audit report completed (Three-tier provenance classification).
[X] 4. Detailed model comparison report completed (Multi-model, Grouped CV & Holdout analysis).
[X] 5. Final comprehensive project report completed.
[X] 6. All 13 automated unit tests passing cleanly.
[X] 7. System kept in baseline demonstration mode awaiting explicit user direction.
========================================================================================
```
