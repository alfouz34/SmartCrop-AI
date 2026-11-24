# 🌱 SmartCrop AI: Intelligent Crop Recommendation & Decision Support System

An advanced agricultural decision-support system built with Python, scikit-learn, FastAPI, and Streamlit. SmartCrop AI analyzes soil nutrient levels and regional meteorological metrics to recommend the top-3 most suitable crops from 22 distinct categories.

---

## 1. Project Overview & Objectives

Agricultural productivity depends heavily on selecting crops well-matched to local soil chemistry and climate dynamics. Incorrect crop choices lead to reduced yields, nutrient depletion, and economic vulnerability for farmers.

### Objectives:
- Train and rigorously benchmark multiple machine learning classifiers using stratified 5-fold cross-validation.
- Select the best-performing model based strictly on training cross-validation macro F1-score to prevent test data leakage.
- Provide transparent model explainability using both built-in Gini feature importance and permutation feature importance.
- Deploy an asynchronous REST API using **FastAPI** with Pydantic input validation.
- Provide an intuitive, interactive dashboard using **Streamlit** and **Plotly** for real-time agricultural decision support.

---

## 2. Dataset Details

The project utilizes the benchmark **Kaggle Crop Recommendation Dataset**:

- **Total Records**: 2,200 samples
- **Features**: 7 numeric input parameters:
  - `N`: Nitrogen level ratio in soil (kg/ha, range: 0 – 140)
  - `P`: Phosphorus level ratio in soil (kg/ha, range: 5 – 145)
  - `K`: Potassium level ratio in soil (kg/ha, range: 5 – 205)
  - `temperature`: Regional temperature (°C, range: 8.8 – 43.7 °C)
  - `humidity`: Relative humidity (%, range: 14.3 – 100.0 %)
  - `ph`: Soil pH value (scale: 3.5 – 9.9)
  - `rainfall`: Precipitation depth (mm, range: 20.2 – 298.6 mm)
- **Target Variable**: `label` (22 crop classes):
  - `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`.
- **Integrity**: Perfectly balanced with exactly 100 samples per class. Verified 0 missing values and 0 duplicate rows.

---

## 3. Technologies Used

- **Language & Runtime**: Python 3.10+ (tested on Python 3.13.9)
- **Machine Learning**: `scikit-learn`, `joblib`, `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Backend API**: `fastapi`, `uvicorn`, `pydantic`
- **Frontend Dashboard**: `streamlit`
- **Testing & Verification**: `pytest`, `httpx2`

---

## 4. Project Folder Structure

```text
SmartCrop_AI/
├── backend/
│   ├── __init__.py
│   └── main.py                     # FastAPI REST API with lifespan preloading
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── Crop_recommendation.csv # Original dataset (2,200 rows)
│       └── PUT_DATASET_HERE.txt
├── frontend/
│   └── app.py                      # Streamlit interactive dashboard & Plotly charts
├── models/
│   ├── best_model.joblib           # Trained Random Forest classifier artifact
│   ├── model_comparison.csv        # 5-fold CV and test benchmark comparison
│   └── model_metadata.json         # Model hyperparameters, classes, and metrics
├── reports/
│   ├── figures/
│   │   ├── correlation_heatmap.png # Feature correlation matrix
│   │   ├── crop_wise_averages.csv  # Tabular per-crop feature means
│   │   ├── crop_wise_averages.png  # Heatmap of crop requirements
│   │   ├── feature_boxplots.png    # IQR outlier inspection boxplots
│   │   ├── feature_distributions.png # Univariate feature histograms
│   │   ├── feature_importance_rf.csv # Built-in Gini importance values
│   │   ├── feature_importance_rf.png # Built-in Gini importance chart
│   │   ├── permutation_importance.csv# Permutation importance metrics
│   │   └── permutation_importance.png# Permutation importance chart
│   └── model_evaluation/
│       ├── classification_report.txt # Per-class precision, recall, and F1
│       └── confusion_matrix.png    # 22x22 confusion matrix heatmap
├── src/
│   ├── __init__.py
│   ├── eda.py                      # Exploratory data analysis pipeline
│   ├── explain.py                  # Gini & permutation feature importance generator
│   ├── predict.py                  # Singleton inference loader & top-3 ranking
│   ├── preprocessing.py            # Dataset loading, validation & stratified split
│   ├── train.py                    # Multi-model CV training & evaluation
│   └── tune.py                     # RandomizedSearchCV hyperparameter tuning
├── tests/
│   ├── test_api.py                 # FastAPI endpoint & validation test suite
│   ├── test_prediction.py          # Unit tests for prediction logic & mocks
│   └── test_preprocessing.py       # Unit tests for data splitting & stratification
├── requirements.txt
└── README.md
```

---

## 5. Data Preprocessing and EDA Findings

- **Data Partitioning**: Stratified 80/20 train/test split (1,760 training samples, 440 test samples). Every crop class maintains an exact 80/20 balance across both splits.
- **Outlier Preservation**: Upper statistical outliers in Phosphorus ($P > 100$), Potassium ($K > 60$), and Rainfall ($> 200\text{ mm}$) represent genuine biological requirements (e.g. apple and grapes require extreme $P$ and $K$; rice requires heavy rainfall). They were strictly retained.
- **Correlation**: Strongest correlation exists between Phosphorus ($P$) and Potassium ($K$) ($r = 0.74$) due to fruit crops requiring both nutrients in high concentration. All other features exhibit low inter-correlation ($|r| < 0.24$), providing distinct, orthogonal signals.

---

## 6. Model Comparison & Selection

Five models were trained and benchmarked using **Stratified 5-Fold Cross-Validation** on the training set. Model selection was finalized strictly using the mean cross-validation Macro F1-score:

| Model Architecture | 5-Fold CV Macro F1 (Mean ± Std) | 5-Fold CV Accuracy | Test Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (300 trees)** | **0.9937 ± 0.0056** | **0.9938** | **0.9932** | **Selected Best Model** |
| Extra Trees (300 trees) | 0.9926 ± 0.0039 | 0.9926 | 0.9955 | Candidate |
| Decision Tree (CART) | 0.9852 ± 0.0068 | 0.9852 | 0.9795 | Candidate |
| SVM (RBF + StandardScaler) | 0.9788 ± 0.0106 | 0.9790 | 0.9841 | Candidate |
| KNN ($k=5$ + StandardScaler)| 0.9650 ± 0.0124 | 0.9653 | 0.9795 | Candidate |

*Random Forest was chosen because its bagged decision trees handle sharp non-linear thresholds and scale variances across nutrients and climate variables effectively.*

---

## 7. Actual Test-Set Evaluation Results

The selected Random Forest model was evaluated on the unseen held-out test split (440 samples):

- **Test Accuracy**: **99.32%** (437 out of 440 correct classifications)
- **Macro Precision**: **0.9935**
- **Macro Recall**: **0.9932**
- **Macro F1-Score**: **0.9932**
- **Per-Class Precision & Recall**: 19 out of 22 crops achieved perfect 1.00 F1-scores. Only 3 instances experienced single-label confusion:
  - 1 Blackgram sample predicted as Lentil
  - 1 Lentil sample predicted as Mothbeans
  - 1 Rice sample predicted as Jute

---

## 8. Model Explainability & Feature Importance

To understand how the Random Forest makes decisions, both built-in impurity importance and test-set permutation importance were computed:

### Actual Calculated Importance Values

| Feature | Built-in Gini Importance (Share) | Permutation Importance (Test Macro F1 Drop) |
| :--- | :---: | :---: |
| **Rainfall** | **22.15%** (0.2215) | **0.1812 ± 0.0130** |
| **Humidity** | **21.19%** (0.2119) | **0.3037 ± 0.0192** |
| **Potassium (K)** | **17.92%** (0.1792) | **0.1575 ± 0.0152** |
| **Phosphorus (P)** | **15.37%** (0.1537) | **0.1134 ± 0.0131** |
| **Nitrogen (N)** | **10.39%** (0.1039) | **0.2023 ± 0.0139** |
| **Temperature** | **7.58%** (0.0758) | **0.0056 ± 0.0034** |
| **Soil pH** | **5.39%** (0.0539) | **0.0098 ± 0.0039** |

### Beginner-Friendly Guide to Feature Importance:
1. **Higher Importance Means Higher Predictive Utility**: In this dataset, rainfall, humidity, and nutrients are the strongest splitters for distinguishing between crops.
2. **Lower Importance Does NOT Mean Unimportant for Agriculture**: Soil pH and temperature are biologically essential for plant survival. However, because most crops in this dataset overlap within standard ranges (pH 5.5–7.5, 20–30°C), they provide fewer distinctive dividing lines between crops than rainfall or humidity extremes.
3. **Correlation is Not Causation**: Feature importance reflects patterns within this specific historical dataset, not biological cause-and-effect.
4. **Why Gini and Permutation Importance Differ**:
   - *Gini Importance* measures how much each feature reduces impurity across all splits in the training trees.
   - *Permutation Importance* measures how much the model's test performance drops when that feature's values are randomly shuffled on unseen test data. In this particular permutation importance experiment, shuffling humidity caused the largest observed macro F1 decrease (0.3037), indicating that model performance on this test split was most sensitive to disruptions in humidity values.

---

## 9. Probability Calibration Status

- **Status**: The production model is **NOT probability-calibrated**.
- **Explanation**: The `RandomForestClassifier.predict_proba()` method computes class probabilities by averaging the class probability estimates produced by the individual decision trees (where each individual tree estimates probabilities based on the fraction of training samples of each class in a leaf).
- **Correct Interpretation**: These scores are **model prediction scores** (uncalibrated model-estimated class probabilities), not calibrated real-world probabilities of farming success.

---

## 10. FastAPI Service & Endpoints

The backend is built with FastAPI and runs with Uvicorn.

- **`GET /health`**: Health check returning service status and model loading state.
- **`POST /predict`**: Accepts JSON input with the 7 parameters, validates numeric bounds using Pydantic, and returns the top 3 recommendations with model prediction scores.
- **Interactive Documentation**: Available at `http://127.0.0.1:8000/docs`.

### Example Prediction Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"N": 90, "P": 42, "K": 43, "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 202.0}'
```

---

## 11. Streamlit Dashboard Features

The web interface provides an accessible UI:
- **System Status**: Displays real-time API connectivity in the sidebar.
- **Preset Scenarios**: One-click loading of realistic profiles (Rice, Apple, Chickpea, Custom).
- **Top 3 Recommendations**: Metric cards displaying ranked crops and percentage model prediction scores.
- **Confidence Chart**: Plotly horizontal/vertical bar charts comparing relative prediction scores.
- **Understanding the Model**: Interactive feature importance chart and educational breakdowns.
- **Agronomic Disclaimer**: Explicit advisory banners informing users that scores are statistical model outputs.

---

## 12. Installation & Setup

1. **Clone or Extract the Project**:
   ```powershell
   cd SmartCrop_AI
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 13. Running the Application

Open two separate PowerShell terminal windows:

### Terminal 1 — Start the FastAPI Backend
```powershell
cd "C:\Users\alfouz\OneDrive\Desktop\SmartCrop_AI_VSCode_Starter\SmartCrop_AI"
$env:PYTHONPATH="."
& "C:\Users\alfouz\anaconda3\python.exe" -m uvicorn backend.main:app --reload --port 8000
```
- API Docs: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

### Terminal 2 — Start the Streamlit Dashboard
```powershell
cd "C:\Users\alfouz\OneDrive\Desktop\SmartCrop_AI_VSCode_Starter\SmartCrop_AI"
$env:PYTHONPATH="."
& "C:\Users\alfouz\anaconda3\python.exe" -m streamlit run frontend/app.py
```
- Dashboard URL: http://localhost:8501

---

## 14. Testing & Verification

Run the full automated test suite using `pytest`:

```powershell
$env:PYTHONPATH="."
& "C:\Users\alfouz\anaconda3\python.exe" -m pytest -v
```

All 13 tests cover:
1. `GET /health` endpoint availability and model loading status.
2. `POST /predict` valid profile predictions (e.g. Rice, Apple).
3. `POST /predict` invalid input validation (negative pH out of range, missing fields).
4. Model prediction logic robustness across array-like and list mocks.
5. Preprocessing shape integrity and stratified class distributions.
6. Preservation and integrity of production artifacts and datasets.
7. Expanded 39-crop schema compliance, row counts, bounds, and provenance tags.

---

## 15. Limitations & Future Improvements

### Current Limitations:
1. **Synthetic / Curated Dataset**: The 2,200 records represent ideal benchmark measurements without real-world sensor noise, seasonal microclimates, or pest pressures.
2. **Missing Economic Variables**: Market demand, crop prices, irrigation costs, and seed accessibility are not incorporated.
3. **No Temporal Dimension**: Historical rainfall distribution across growth stages (e.g. vegetative vs. flowering) is aggregated into a single precipitation value.

### Future Improvements:
- Integrate live weather APIs (e.g. OpenWeatherMap) to auto-fill climate data.
- Incorporate geospatial location mapping for regional soil suitability.
- Implement post-hoc probability calibration (e.g. Isotonic Regression via `CalibratedClassifierCV`) if true real-world risk probabilities are required.

---

## ⚠️ Agricultural Decision-Support Disclaimer

**Important**: SmartCrop AI provides statistical decision-support recommendations based on machine learning pattern recognition. It does **not** provide guaranteed agricultural advice or farming instructions. Model prediction performance on this dataset does not guarantee real-world crop yield or farming success. Farmers and agronomists must verify recommendations against local soil testing labs, regional extension services, and seasonal weather forecasts.
