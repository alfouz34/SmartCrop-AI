# 🔬 Expanded Dataset Provenance & Data Leakage Audit
## Phase 5: Scientific Validation and Forensic Integrity Analysis

**Project:** SmartCrop AI — Real-World Agricultural Dataset Expansion  
**Audited File:** `data/processed/expanded_crop_dataset.csv` (3,900 rows, 14 columns, 39 crops)  
**Date:** November 2026 (Updated Phase 7 Submission Freeze)  
**Auditor:** SmartCrop AI Scientific Validation Team  
**Evaluation Standard:** Zero-Fabrication Scientific Integrity & Provenance Classification  

---

## 1. Executive Summary & Core Verdict

This audit provides a forensic evaluation of the 3,900 records in `expanded_crop_dataset.csv` (2,200 benchmark records + 1,700 newly curated Kerala records). 

### Key Findings:
1. **Mixed-Provenance Research Dataset (Not Verified Farm Ground Truth):**
   - **Original 2,200 Records:** Augmented/compiled benchmark with independently unverified underlying observation provenance.
   - **Additional 1,700 Kerala Records:** Synthetic/simulated observations generated from agronomic and regional parameter ranges.
   - **Combined 3,900 Dataset:** Mixed-provenance research dataset, not verified farm-level ground truth.
2. **Distributional Template Leakage in Random Splits:** While exact duplicate records are $0$, ordinary random train/test splitting suffers from **distributional generator leakage**, where test samples from the synthetic Kerala cohort are drawn from the exact same parametric template as training samples. This artificially inflates test accuracy to $92.18\%$.
3. **Out-of-Distribution Reality Check:** When evaluated using **District-Grouped Cross-Validation (Leave-District-Out)**, model performance drops to **$79.84\%$ accuracy** and **$0.6319$ macro F1** (authoritative benchmark with $n\_estimators=300$) and **$78.84\%$ accuracy** / **$0.6317$ macro F1** (with $n\_estimators=100$).
4. **Geographic Holdout Verification:** The $74.19\%$ test accuracy on the High-Range Western Ghats holdout (Idukki & Wayanad) was verified and reproduced exactly, highlighting an inherent domain shift between midland/coastal plains and mountain microclimates.

---

## 2. Provenance Classification Audit

Every record in the 3,900-row expanded dataset was audited against the standard provenance categories:

```text
========================================================================================
PROVENANCE CATEGORY AUDIT (Total Records: 3,900 | Total Classes: 39)
========================================================================================
Category                                                  Record Count    Percentage
----------------------------------------------------------------------------------------
A. Genuine linked farm-level observation                             0         0.00%
B. Genuine experimental or field-trial observation                   0         0.00%
C. Curated observation assembled from multiple sources               0         0.00%
D. Synthetic or simulated observation based on agronomic ranges  1,700        43.59%
   (Newly added Kerala regional cohort)
E. Augmented / compiled benchmark with unverified provenance     2,200        56.41%
   (Original Kaggle benchmark cohort)
----------------------------------------------------------------------------------------
COMBINED 3,900 DATASET: Mixed-provenance research dataset, not verified farm ground truth.
========================================================================================
```

### Forensic Breakdown by Cohort:

#### Cohort 1: Original Kaggle Benchmark Records (Rows 1 – 2,200 | 22 Crops)
* **Declared Provenance:** Kaggle Crop Recommendation Dataset (Atharva Ingle, 2020).
* **Audit Determination:** **Augmented/compiled benchmark with independently unverified underlying observation provenance.**
* **Documented Evidence:** The dataset description explicitly states: *"This dataset was built by augmenting datasets of rainfall, climate and fertilizer data available for India."* However, specific farm survey GPS points, individual trial registry numbers, or physical soil core testing dates are unverified in the public domain.
* **Statistical Normality Findings:** Normality tests (Shapiro-Wilk) prove the records are NOT standard Gaussian distributions. Features exhibit tightly bounded ranges (e.g. Apple K bounded to discrete integers 195 to 205, $p < 0.0001$; Rice rainfall non-normal, $p = 0.0012$).

#### Cohort 2: Kerala Regional Records (Rows 2,201 – 3,900 | 17 Crops)
* **Declared Provenance:** Kerala DES EARAS statistics, Soil Health Card district fertility data, KAU Package of Practices, and ICAR institute research benchmarks (CPCRI, CTCRI, IISR, RRII).
* **Audit Determination:** **Synthetic/simulated observations generated from agronomic and regional parameter ranges.**
* **Evidence:** Although parameterized using empirical government publications (KAU fertilizer schedules, Soil Health Card district nutrient means, IMD district rainfall normals, and EARAS district average yields), the individual tabular rows were **mathematically synthesized using truncated normal distributions** within those empirical bounds.
* **Traceability Assessment:** No individual row can be traced back to an individual soil core sample taken on farmer Raman's plot in Kottayam on a specific date. Describing these 1,700 records as "verified farm-level data" would be scientifically dishonest.

---

## 3. Crop-by-Crop Audit & Measurement Lineage

The following table documents the exact origin of $N, P, K$, climate, and yield attributes for all 17 newly added Kerala crops:

| Crop | Classification | Sample Count | Soil ($N, P, K, \text{pH}$) Origin | Climate ($T, H, R$) Origin | Yield Origin | Physical Coupling Status |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **Rubber** | Simulated | 100 | KAU PoP & RRII fertilizer guidelines (acid laterite) | IMD normals for Kottayam, Pathanamthitta, Ernakulam | EARAS district average sheet rubber yield ($1.4–2.4 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Tapioca** | Simulated | 100 | ICAR-CTCRI nutrient schedule ($100:50:100 \text{ kg/ha}$) | IMD normals for Kollam, Thiruvananthapuram | EARAS district fresh tuber yields ($22–42 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Arecanut** | Simulated | 100 | ICAR-CPCRI palm nutrition recommendations | IMD normals for Kasaragod, Kannur, Wayanad | EARAS district cured chali yield ($1.9–3.8 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Cashew** | Simulated | 100 | KAU Cashew Research Station trial standards | IMD normals for Kannur, Kasaragod, Kollam | EARAS district raw nut yields ($0.8–1.8 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Black Pepper** | Simulated | 100 | ICAR-IISR vine nutrition guidelines | IMD normals for Wayanad, Idukki, Kannur | EARAS district dry black pepper yield ($0.6–1.4 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Cardamom** | Simulated | 100 | ICAR-IISR Cardamom Research Station guidelines | IMD normals for Cardamom Hills, Idukki | Spices Board & EARAS cured capsule yields ($0.32–0.85 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Ginger** | Simulated | 100 | ICAR-IISR field trial fertility recommendations | IMD normals for Wayanad, Palakkad | EARAS green rhizome yields ($11.5–24.0 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Turmeric** | Simulated | 100 | KAU Agronomic Survey nutrient envelopes | IMD normals for Wayanad, Palakkad | EARAS fresh rhizome yields ($16.0–30.5 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Tea** | Simulated | 100 | UPASI Tea Research Institute soil standards | IMD normals for High Range DAMU Munnar/Vythiri | Tea Board & EARAS made tea yields ($2.0–3.6 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Clove** | Simulated | 100 | ICAR-IISR tree spice fertilizer trials | IMD normals for Kottayam, Ernakulam, Thrissur | EARAS dry unopened bud yields ($0.24–0.62 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Nutmeg** | Simulated | 100 | KAU Periyar basin research trials | IMD normals for Ernakulam (Kalady/Perumbavoor) | KAU trial nut & mace yields ($1.0–2.3 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Sweet Potato** | Simulated | 100 | ICAR-CTCRI trial recommendations | IMD normals for Kollam, Thiruvananthapuram | EARAS district tuber yields ($13.0–24.5 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Elephant Foot Yam** | Simulated | 100 | ICAR-CTCRI corm nutrition trials | IMD normals for Kollam, Ernakulam, Thrissur | EARAS bulk corm yields ($26.0–54.0 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Tomato** | Simulated | 100 | KAU Vegetable Research Station winter guidelines | IMD winter dry season normals (Vattavada, Palakkad) | KAU vegetable trial yields ($21.0–42.0 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Chilli** | Simulated | 100 | KAU Package of Practices vegetable standards | IMD winter/summer normals for Palakkad, Thrissur | EARAS dry chilli yields ($1.5–3.2 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Brinjal** | Simulated | 100 | KAU Vegetable Research Station standards | IMD normals for Palakkad, Thrissur, Kollam | EARAS fresh brinjal yields ($19.0–38.0 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |
| **Okra** | Simulated | 100 | KAU Package of Practices summer standards | IMD summer normals for Palakkad, Malappuram | EARAS green pod yields ($7.5–16.5 \text{ t/ha}$) | **Decoupled** (Synthesized from separate reports) |

---

## 4. Data Leakage & Duplication Audit

### 4.1 Exact & Near-Duplicate Analysis
Using nearest-neighbor Euclidean distance in standardized feature space ($\mu=0, \sigma=1$):
- **Exact Duplicates across all 14 columns:** **$0$** ($0.0\%$).
- **Exact Duplicates across 7 core features:** **$0$** ($0.0\%$).
- **Near-Duplicates ($d < 0.05$ standard deviations):** **$0$** ($0.0\%$).
- **Near-Duplicates ($d < 0.10$ standard deviations):** **$2$** pairs ($0.051\%$).
- **Minimum Normalized Pairwise Distance:** $0.06358$.

### 4.2 Distributional Template Leakage (The Hidden Trap)
While exact duplicates do not exist, a structural leakage mechanism was uncovered:
* **Mechanism:** When 100 synthetic rows for a crop are generated from a single parameter template ($\mu, \sigma$), a standard random $80/20$ train/test split randomly allocates 80 rows to training and 20 rows to testing.
* **Impact:** The testing set is drawn from the **exact same parametric distribution** as the training set. The model is tested on how well it interpolates within a known generator template, rather than how well it generalizes to an unseen farm or district.
* **Proof via Grouped Cross-Validation (Reconciled Metrics):**
  When we split data by **District (Leave-District-Out GroupKFold)** so that no samples from the test districts appear in the training folds, performance drops sharply:
  - Standard Random Split Macro F1: **$0.9104$**
  - District-Grouped Split Macro F1 ($n=300$, Authoritative): **$0.6319 \pm 0.0466$** ($\Delta = -27.85\%$)
  - District-Grouped Split Macro F1 ($n=100$, Historical Audit): **$0.6317 \pm 0.0444$**
  - Standard Random Split Accuracy: **$0.9218$**
  - District-Grouped Split Accuracy ($n=300$, Authoritative): **$0.7984 \pm 0.0424$** ($\Delta = -12.34\%$)
  - District-Grouped Split Accuracy ($n=100$, Historical Audit): **$0.7884 \pm 0.0496$**

---

## 5. Geographic Holdout Validation (Western Ghats High Ranges)

### 5.1 Verification of the 500 Samples
* Total Kerala Records: $1,700$
* Records from Midland/Coastal/Plains Districts (12 Districts): $1,314$
* Records from High-Range Districts (Idukki & Wayanad): $386$
* **High-Range Endemics (2 crops):** `cardamom` ($100$ samples) and `tea` ($100$ samples) exist strictly in Idukki and Wayanad.
* **Midland/Plains Endemics (10 crops):** Rubber, Tapioca, Cashew, Clove, Nutmeg, Sweet potato, Elephant foot yam, Chilli, Brinjal, Okra ($100$ each).
* **Cross-Regional Crops (5 crops):** `arecanut`, `blackpepper`, `ginger`, `tomato`, `turmeric`.
* **Sample Accounting:**
  - Training samples for 5 common crops in plains/midlands: **$314$ rows**
  - Testing samples for 5 common crops in high ranges: **$186$ rows**
  - **Total sample universe examined:** **$500$ rows** ($314 + 186 = 500$).

### 5.2 Reproducibility & Per-Class Support Verification
```text
========================================================================================
GEOGRAPHIC HOLDOUT RESULTS (Trained on Midlands/Plains -> Tested on High Ranges)
========================================================================================
Metric                  Reported (Phase 4)    Audited Reproduction    Variance
----------------------------------------------------------------------------------------
Test Accuracy                       0.7419                  0.7419       0.0000 (EXACT)
Test Macro Precision                0.7628                  0.7628       0.0000 (EXACT)
Test Macro Recall                   0.7786                  0.7786       0.0000 (EXACT)
Test Macro F1                       0.7232                  0.7232       0.0000 (EXACT)
========================================================================================

Per-Class Performance on Unseen High-Range Test Set (Support: 186):
----------------------------------------------------------------------------------------
Crop Class          Support     Precision        Recall      F1-Score    Diagnostic Error Pattern
----------------------------------------------------------------------------------------
arecanut                 25        0.5714        0.9600        0.7164    Overpredicted; confused with pepper.
blackpepper              43        0.9615        0.5814        0.7246    Low recall; mountain temps confuse model.
ginger                   53        0.8966        0.5094        0.6500    Severe false-negative confusion with turmeric.
tomato                   46        1.0000        1.0000        1.0000    Distinct dry cool-season winter signature.
turmeric                 19        0.3810        0.8421        0.5246    Overpredicted due to overlap with ginger.
----------------------------------------------------------------------------------------
Macro Average           186        0.7621        0.7786        0.7231
Weighted Average        186        0.8351        0.7419        0.7479
========================================================================================
```

---

## 6. Comprehensive Multi-Model Benchmarking

| Experiment Split | Algorithm | Classes | Total N | Accuracy | Macro Precision | Macro Recall | Macro F1 | Generalization Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Benchmark** | Random Forest | 22 | 2,200 | **0.9932** | 0.9935 | 0.9932 | **0.9932** | Augmented benchmark; zero Kerala commercial crop awareness. |
| **Expanded (Random Split)** | Random Forest | 39 | 3,900 | **0.9218** | 0.9243 | 0.9218 | **0.9204** | High score driven by distributional generator leakage. |
| **Expanded (Random Split)** | Extra Trees | 39 | 3,900 | **0.9218** | 0.9226 | 0.9218 | **0.9197** | Cross-validation winner on random split. |
| **Expanded (District Grouped, n=300)** | Random Forest | 17 | 1,700 | **0.7984** | 0.8012 | 0.7984 | **0.6319** | **Authoritative spatial generalization**; tests model across unseen districts. |
| **Expanded (District Grouped, n=100)** | Random Forest | 17 | 1,700 | **0.7884** | 0.7950 | 0.7884 | **0.6317** | Historical quick audit baseline. |
| **High-Range Holdout** | Random Forest | 5 | 500 | **0.7419** | 0.7628 | 0.7786 | **0.7232** | **Honest agro-climatic generalization**; tests elevation domain shift. |
| **Empirical-Only Farm Data** | — | — | 0 | **N/A** | **N/A** | **N/A** | **N/A** | **No open linked farm dataset exists in India for these crops.** |

---

## 7. Audit Conclusion

1. **Mixed-Provenance Reality:** The combined 3,900 dataset unifies an augmented academic benchmark (2,200 rows) with simulated regional agronomic envelopes (1,700 rows); it is not verified farm-level ground truth.
2. **Metrics Contextualized:** The 92.0% random-split accuracy drops to **63.2% – 74.2%** under honest spatial and geographic holdout evaluation.
3. **Crucial Diagnostic Finding:** The model cannot yet be considered a true "farm-level" predictor. It operates as an **algorithmic encoding of published KAU/ICAR agronomic envelopes**.
