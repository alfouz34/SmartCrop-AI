# 📊 Model Comparison & Agricultural Limitations Report
## Phase 6 Technical Analysis: Model Benchmarking, Holdout Verification & Diagnostic Limitations

**Project:** SmartCrop AI — Real-World Agricultural Dataset Expansion  
**Report Reference:** `reports/model_comparison_and_limitations.md`  
**Date:** November 2026 (Updated Phase 7 Submission Freeze)  
**Status:** Complete Technical Report  

---

## 1. Comprehensive Model Comparison Matrix

The following table summarizes all model experiments conducted, documenting sample sizes, validation frameworks, metrics, and generalization characteristics:

```text
============================================================================================================================================
COMPREHENSIVE MODEL COMPARISON BENCHMARK TABLE
============================================================================================================================================
Experiment Identifier                 Classes  Total N  CV Metric        CV Macro F1  Test Acc  Test Prec  Test Rec  Test F1  Status & Purpose
--------------------------------------------------------------------------------------------------------------------------------------------
Random Forest (Original Baseline)          22    2,200  Stratified 5-CV       0.9937    0.9932     0.9935    0.9932   0.9932  ACTIVE PRODUCTION
Extra Trees (Expanded)                     39    3,900  Stratified 5-CV       0.9122    0.9218     0.9226    0.9218   0.9197  CV Winner (39 Crops)
Random Forest (Expanded)                   39    3,900  Stratified 5-CV       0.9104    0.9218     0.9243    0.9218   0.9204  Ensemble Experimental
SVM (RBF Kernel, Scaled)                   39    3,900  Stratified 5-CV       0.8893    0.8897     0.8931    0.8897   0.8885  Kernel Experimental
Decision Tree (Expanded)                   39    3,900  Stratified 5-CV       0.8558    0.8705     0.8720    0.8705   0.8691  Single Tree Baseline
KNN (k=5, Scaled)                          39    3,900  Stratified 5-CV       0.8561    0.8679     0.8721    0.8679   0.8670  Instance Baseline
District-Grouped CV (n=300, Authoritative) 17    1,700  5-Fold GroupKFold     0.6319    0.7984     0.8012    0.7984   0.6319  Spatial Benchmark
District-Grouped CV (n=100, Hist. Audit)   17    1,700  5-Fold GroupKFold     0.6317    0.7884     0.7950    0.7884   0.6317  Historical Audit
Geographic Holdout (Random Forest)          5      500  High-Range Split         N/A    0.7419     0.7628    0.7786   0.7232  Mountain Domain Shift
============================================================================================================================================
```

---

## 2. Metric Reconciliation: District-Grouped Cross-Validation

During Phase 5 and Phase 6 analysis, a minor metric variance was recorded for District-Grouped Cross-Validation:
- **Phase 5 Quick Audit Metric:** Accuracy = **$0.7884$**, Macro F1 = **$0.6317$**.
- **Phase 6 Authoritative Metric:** Accuracy = **$0.7984$**, Macro F1 = **$0.6319$**.

### The Technical Root Cause:
The variance was traced to the ensemble tree hyperparameter (`n_estimators`):
1. **$n\_estimators = 100$ (Phase 5 Rapid Audit):** Evaluated using a lightweight 100-tree forest for fast preliminary screening.
   - Mean Accuracy: $0.7884 \pm 0.0496$
   - Mean Macro F1: $0.6317 \pm 0.0444$
2. **$n\_estimators = 300$ (Phase 6 Authoritative Benchmark):** Evaluated using the standard 300-tree forest matching all candidate models in `src/train_expanded.py`.
   - Mean Accuracy: $0.7984 \pm 0.0424$
   - Mean Macro F1: $0.6319 \pm 0.0466$

Both evaluations use the identical 1,700 Kerala records, identical 14-district grouping, and `random_state=42`. For formal project reporting, **$n\_estimators=300$ ($0.7984 / 0.6319$) is declared the authoritative benchmark**.

---

## 3. Forensic Audit of the Geographic Holdout Experiment

### 3.1 Sample Accounting & Selection Methodology ($N=500$)
1. **Total Kerala Dataset:** 1,700 records (17 crops $\times$ 100 samples).
2. **Geographic Partition:**
   - **Midland, Coastal & Plains Districts (12 Districts):** Thiruvananthapuram, Kollam, Pathanamthitta, Alappuzha, Kottayam, Ernakulam, Thrissur, Palakkad, Malappuram, Kozhikode, Kannur, Kasaragod $\rightarrow$ **$1,314$ total records**.
   - **High-Range Western Ghats Districts (2 Districts):** Idukki and Wayanad $\rightarrow$ **$386$ total records**.
3. **The Endemic Asymmetry Constraint:**
   - Two crops in the dataset (`cardamom` and `tea`) are **strictly endemic to the high ranges** ($100$ samples each, all located in Idukki and Wayanad). Because they have zero observations in the plains/midlands, a classifier trained exclusively on non-holdout districts has never observed them; testing on them would mathematically produce $0\%$ recall.
   - Ten crops (`rubber`, `tapioca`, `cashew`, `clove`, `nutmeg`, `sweetpotato`, `elephantfootyam`, `chilli`, `brinjal`, `okra`) exist exclusively in the midlands and plains.
   - Only **5 crops co-occur across both regions**: `arecanut`, `blackpepper`, `ginger`, `tomato`, `turmeric`.
4. **Final Holdout Cohort ($N=500$):**
   - Training split ($N=314$): The 5 common crops in the 12 midland/coastal/plain districts.
   - Testing split ($N=186$): The 5 common crops in the 2 high-range districts (Idukki and Wayanad).
   - **Total sample universe:** $314 + 186 = \mathbf{500\text{ samples}}$.

### 3.2 Ground-Truth Nature: Observed vs. Simulated
- The `district` assignments and crop labels were **assigned during data synthesis** based on published cultivating districts from KAU and EARAS reports.
- The feature values (temperature, rainfall, pH) were **simulated** using truncated normal distributions around the district's published climatic and soil rating baselines.
- **Scientific Determination:** This experiment **does NOT represent physical geographic generalization across real-world farms**; it evaluates the mathematical generalization between two simulated regional distributions parameterized on differing agro-climatic envelopes.

### 3.3 Detailed Per-Class Holdout Metrics (Reproduced: Accuracy $74.19\%$, Macro F1 $0.7232$):

```text
========================================================================================================
GEOGRAPHIC HOLDOUT PER-CLASS DIAGNOSTIC TABLE (Test Samples: 186 | Train Samples: 314)
========================================================================================================
Crop Class     Train Support  Test Support  Precision  Recall  F1-Score  Diagnostic Error Pattern
--------------------------------------------------------------------------------------------------------
arecanut                  75            25     0.5714  0.9600    0.7164  High recall; false positives from pepper.
blackpepper               57            43     0.9615  0.5814    0.7246  Low recall; mountain cold confuses model.
ginger                    47            53     0.8966  0.5094    0.6500  High false negatives; confused with turmeric.
tomato                    54            46     1.0000  1.0000    1.0000  Distinct winter cool-dry season signature.
turmeric                  81            19     0.3810  0.8421    0.5246  Overpredicted; severe confusion with ginger.
--------------------------------------------------------------------------------------------------------
Macro Average            314           186     0.7621  0.7786    0.7231  Reproduced Exactly (0.7232)
Weighted Avg             314           186     0.8351  0.7419    0.7479  Accuracy: 74.19% (Reproduced)
========================================================================================================
```

---

## 4. Deep Analysis of Agricultural & Real-World Limitations

### 4.1 Mixed-Provenance Research Reality
Neither the original Kaggle benchmark nor the newly expanded Kerala dataset contains verified on-farm empirical measurements. The dataset unifies an augmented academic benchmark (2,200 rows) with simulated regional agronomic envelopes (1,700 rows). While valuable for prototyping and educational software, it cannot guarantee real-world biological outcomes.

### 4.2 Topographic Blindness (Missing Elevation & Slope)
In Kerala's undulating terrain (stretching from sea level in Kuttanad to $>2,600\text{ m}$ in the Western Ghats), elevation is a primary bioclimatic filter:
- **Cardamom:** Strictly bounded to $800–1,400\text{ m}$ MSL.
- **Tea:** Strictly bounded to $1,000–2,200\text{ m}$ MSL.
- **Rubber:** Unsuitable above $600\text{ m}$ MSL due to delayed growth and tapping panel cold-injury.
Without an explicit `elevation` feature, the model relies entirely on temperature as a proxy, leading to boundary confusion in transitional foothill zones.

### 4.3 Hydrological Blindness (Missing Drainage & Soil Depth)
Kerala receives $3,000\text{ mm}$ of precipitation annually. The primary agronomic constraint for high-value crops is not water scarcity, but **waterlogging and soil aeration**:
- **Black Pepper:** Highly susceptible to Quick Wilt (*Phytophthora capsici*) if soil drainage is poor, even if NPK and rainfall values are optimal.
- **Rubber:** Requires deep soils ($>1\text{ m}$) with good internal drainage; shallow soils cause uprooting during monsoon winds.
The current 7-feature schema does not model soil drainage class, water table depth, or soil depth.

### 4.4 Capital Risk Asymmetry in Kerala Perennial Farming
- In annual grain farming (e.g. Rice, Maize), an incorrect crop choice costs 3 to 4 months of operational expenditure.
- In Kerala plantation farming, planting Rubber or Arecanut represents an irreversible **5 to 7 year capital commitment** before the first yield. An unverified AI recommendation that fails due to drainage or elevation causes devastating long-term financial insolvency for a smallholder farmer.

### 4.5 Polyculture vs. Monoculture Mismatch
Real Kerala farms are multi-tier homestead polycultures (Coconut canopy + Pepper vine on Arecanut + understory of Tapioca and Ginger). The current single-class classifier forces a mutually exclusive prediction, which is architecturally misaligned with sustainable agroforestry practices in Kerala.
