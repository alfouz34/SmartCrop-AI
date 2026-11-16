# 🌾 Agricultural Data Sources & Investigation Report
## Real-World Datasets for Kerala and India Crop Recommendation Systems

**Project:** SmartCrop AI — Real-World Agricultural Dataset Expansion  
**Focus Region:** Kerala, South India & All-India Agricultural Ecosystem  
**Date:** November 2026  
**Author:** SmartCrop AI Research & Data Science Engineering  

---

## 1. Executive Summary & Problem Context

The original SmartCrop AI prototype was developed using the widely shared 22-crop **Kaggle Crop Recommendation Dataset** (compiled by Atharva Ingle). While this dataset provides a standardized benchmark for machine learning algorithms (achieving >99% macro F1-score with ensemble classifiers), it exhibits three severe real-world shortcomings when applied to Indian farming, particularly in the State of Kerala:

1. **Synthetic Feature Distributions:** The original benchmark features Gaussian distributions artificially generated around agronomic requirement centroids. As a result, standard machine learning classifiers experience near-zero boundary overlap, producing deceptively optimistic test accuracies that do not reflect farm-level operational realities.
2. **Missing Regional Crop Diversity:** Kerala is India's leading producer of plantation and spice crops. Major commercial drivers of Kerala's agricultural economy—including **Rubber (*Hevea brasiliensis*)**, **Tapioca (*Manihot esculenta*)**, **Arecanut (*Areca catechu*)**, **Cashew (*Anacardium occidentale*)**, **Black Pepper (*Piper nigrum*)**, **Cardamom (*Elettaria cardamomum*)**, **Ginger (*Zingiber officinale*)**, **Turmeric (*Curcuma longa*)**, **Tea (*Camellia sinensis*)**, **Nutmeg (*Myristica fragrans*)**, and tropical vegetables (**Tomato, Chilli, Brinjal, Okra**)—are completely absent from the 22-class benchmark.
3. **The Micro vs. Macro Data Dilemma (The Ecological Fallacy):** Publicly available government agricultural statistics are heavily aggregated at the **district level** (macro-level Area, Production, and Yield), whereas crop recommendation algorithms require **field-level / farm-level** soil chemistry ($N, P, K, \text{pH}$) and localized microclimate ($temperature, humidity, rainfall$).

This report provides a scientifically honest audit of official government repositories, research institute datasets, and public platforms. For every source investigated, we record official URLs, geographic coverage, time horizon, available parameters, data collection methodologies, and **explicit limitations**.

---

## 2. Priority Source Investigations

### 2.1 Kerala Department of Economics and Statistics (DES Kerala) — EARAS
* **Source Organization:** Department of Economics and Statistics (DES), Government of Kerala, in collaboration with the Directorate of Economics and Statistics (DES India), Ministry of Agriculture & Farmers Welfare.
* **Official URL / Portal:** [https://ecostat.kerala.gov.in](https://ecostat.kerala.gov.in) | Open Data Module: [http://117.241.78.118](http://117.241.78.118)
* **Key Publications:**
  - *Establishment of an Agency for Reporting Agricultural Statistics (EARAS) Annual Survey Reports*
  - *Agricultural Statistics Compendium, Kerala* (published quinquennially / annually)
  - *Statistics for Planning / Economic Review (Agriculture & Allied Sectors)*
* **Geographic Coverage:** All 14 administrative districts of Kerala (Thiruvananthapuram, Kollam, Pathanamthitta, Alappuzha, Kottayam, Idukki, Ernakulam, Thrissur, Palakkad, Malappuram, Kozhikode, Wayanad, Kannur, Kasaragod), spanning 61 Taluks and selected representative revenue villages.
* **Time Horizon:** 1975–76 to present (annual time series; latest consolidated reports through 2022–2025).
* **Available Columns & Metrics:**
  - `State`, `District`, `Taluk`
  - `Agricultural Year`, `Season` (Autumn / *Virppu*, Winter / *Mundakan*, Summer / *Puncha*)
  - `Crop Name` (Food grains, Tubers, Plantations, Spices, Fruits, Vegetables, Oilseeds)
  - `Area under Cultivation` (Hectares)
  - `Total Production` (Metric Tonnes, or nuts in millions for Coconut/Arecanut)
  - `Yield / Productivity` (Kilograms per Hectare, or nuts per tree/ha)
  - `Land Utilization Statistics` (Net sown area, fallow land, barren land, forest cover)
  - `Irrigated vs. Unirrigated Cropped Area`
* **Methodology:** Stratified two-stage random sampling design. Revenue villages form the first-stage units; agricultural clusters/survey numbers form the second-stage units. Field investigators conduct physical crop-cutting experiments (CCEs) across designated sample plots ($5 \text{ m} \times 5 \text{ m}$ cuts for paddy, cluster counts for tree crops) to objectively compute mean productivity.
* **Critical Limitations for ML Crop Recommendation:**
  > [!WARNING] **Aggregated Macro Statistics Only (No Soil Chemistry):**
  > EARAS survey records report district and taluk aggregations of **Area, Production, and Yield (APY)**. The survey **does not measure or record farm-level soil macronutrients ($N, P, K$), micronutrients, or soil $\text{pH}$**. It also does not record field-level microclimate (temperature, relative humidity). 
  > **Scientific Integrity Rule:** EARAS district yields cannot be converted into individual farm soil observations without committing an ecological fallacy. They serve as ground-truth regional productivity baselines, validation benchmarks, and season/district mapping references.

---

### 2.2 Government of India Open Data Platform (data.gov.in / UPAg / APY Portal)
* **Source Organization:** Directorate of Economics and Statistics (DES-Agri), Department of Agriculture & Farmers Welfare (DA&FW), Ministry of Agriculture and Farmers Welfare, Government of India.
* **Official URL / Portal:** [https://data.gov.in](https://data.gov.in) | [https://upag.gov.in](https://upag.gov.in) (Unified Portal for Agricultural Statistics) | [https://data.desagri.gov.in](https://data.desagri.gov.in)
* **Dataset Identifier:** *District-wise, season-wise crop production statistics (APY)* (formerly published on data.gov.in, now integrated into UPAg).
* **Geographic Coverage:** Pan-India coverage across 36 States and Union Territories, spanning >700 districts, including all 14 districts of Kerala.
* **Time Horizon:** 1997–98 to 2021–22 (continuous annual time series).
* **Available Columns & Metrics:**
  - `State_Name` (e.g., Kerala)
  - `District_Name` (e.g., Kottayam, Wayanad, Palakkad)
  - `Crop_Year` (e.g., 2019, 2020)
  - `Season` (`Kharif`, `Rabi`, `Summer`, `Whole Year`, `Autumn`, `Winter`)
  - `Crop` (>100 crop varieties including Rice, Coconut, Rubber, Arecanut, Cashewnut, Tapioca, Black pepper, Cardamom, Banana, Tea, Coffee, Turmeric, Ginger, Sweet potato)
  - `Area` (Hectares)
  - `Production` (Tonnes)
  - `Yield` ($\text{Production} / \text{Area}$, metric tonnes/ha or kg/ha)
* **Methodology:** Compiled through the General Crop Estimation Survey (GCES) conducted by State Agricultural Statistics Authorities (SASAs) and State DES bureaus under central supervision.
* **Critical Limitations for ML Crop Recommendation:**
  > [!IMPORTANT] **Macro-Economic Yield vs. Agronomic Input Gap:**
  > Like EARAS, this national dataset contains zero soil test values ($N, P, K, \text{pH}$) and zero meteorological measurements ($temperature, humidity, rainfall$). Missing values exist in production numbers where crop-cutting experiments were not feasible or areas were negligible (<10 ha). Units must be standardized (e.g., coconut production is recorded in nuts/bales in raw tables, whereas food grains are in metric tonnes).

---

### 2.3 Kerala Agriculture Department (Karshika Keralam & Farm Information Bureau)
* **Source Organization:** Department of Agriculture Development and Farmers' Welfare, Government of Kerala.
* **Official URL / Portal:** [https://keralaagriculture.gov.in](https://keralaagriculture.gov.in) | [https://karshikakeralam.gov.in](https://karshikakeralam.gov.in) | [https://fibkerala.gov.in](https://fibkerala.gov.in)
* **Key Publications:**
  - *Farm Guide* (Annual compendium of crop calendars, input costs, and recommended cultivars)
  - *Package of Practices Crop Advisory Bulletins*
  - *AIMS Portal (Agricultural Information Management System)*
* **Geographic Coverage:** Krishi Bhavans across all 1,034 Grama Panchayats, Municipalities, and Corporations in Kerala.
* **Time Horizon:** Annual advisories (1970–present).
* **Available Columns & Metrics:**
  - Regional crop suitability calendars
  - Recommended varieties per agro-ecological unit (AEU 1 through AEU 23)
  - Sowing/planting windows (Virppu, Mundakan, Puncha)
  - Recommended basal and top-dressing nutrient schedules ($\text{kg/ha}$ of Urea, Rajphos/SSP, and Muriate of Potash)
  - Lime / dolomite soil amelioration recommendations for acidic soils ($\text{pH} < 5.5$)
  - Subsidy and crop insurance yield cut-offs
* **Critical Limitations for ML Crop Recommendation:**
  - Qualitative and advisory format rather than tabular point-source machine learning matrices.
  - Recommended nutrient schedules represent agronomist-prescribed inputs needed to achieve target yields, not natural background soil nutrient measurements.

---

### 2.4 National Soil Health Card (SHC) Portal (DAC&FW, Govt. of India)
* **Source Organization:** Department of Agriculture & Farmers Welfare, Ministry of Agriculture & Farmers Welfare, Government of India.
* **Official URL / Portal:** [https://soilhealth.dac.gov.in](https://soilhealth.dac.gov.in) | Open Data mirror on Dataful: [https://dataful.in](https://dataful.in) (*District and Village wise Soil Health Information for Kerala*)
* **Geographic Coverage:** Pan-India; Kerala coverage includes all 14 districts down to village / block levels (over 1.5 million soil sample records collected across Cycles 1 and 2, 2015–2019, and ongoing model village schemes).
* **Time Horizon:** 2015–present.
* **Available Columns & Metrics:**
  - `State`, `District`, `Block / Taluk`, `Village`
  - `Available Nitrogen (N)`: Alkaline permanganate method ($\text{kg/ha}$; rated as Low $<280$, Medium $280–560$, High $>560$)
  - `Available Phosphorus (P)`: Bray No. 1 method for acid soils ($\text{kg/ha}$; rated as Low $<10$, Medium $10–25$, High $>25$)
  - `Available Potassium (K)`: Neutral normal ammonium acetate method ($\text{kg/ha}$; rated as Low $<120$, Medium $120–280$, High $>280$)
  - `Soil Reaction (pH)`: 1:2.5 soil:water suspension (rated as Strongly Acidic $<4.5$, Moderately Acidic $4.5–5.5$, Slightly Acidic $5.5–6.5$, Neutral $6.5–7.5$, Alkaline $>7.5$)
  - `Electrical Conductivity (EC)` ($\text{dS/m}$)
  - `Organic Carbon (OC)` (% content; Walkley-Black method)
  - `Micronutrients`: Sulphur ($S$), Zinc ($Zn$), Iron ($Fe$), Copper ($Cu$), Manganese ($Mn$), Boron ($B$) in $\text{ppm}$ / $\text{mg/kg}$
* **Methodology:** Grid-based soil sampling (2.5 ha grid in irrigated areas, 10 ha grid in rainfed areas). Soil testing performed by District Soil Testing Laboratories (DSTLs) and Mobile Soil Testing Laboratories.
* **Critical Limitations for ML Crop Recommendation:**
  > [!CAUTION] **Independent Soil Census Unlinked to Crop Outcomes:**
  > While the Soil Health Card scheme is the most comprehensive source of empirical soil test values in India, the public portal records soil chemistry independently of continuous crop yield outcomes and local weather time series. It records what was currently planted on the sample date, but does not provide paired seasonal yield or daily weather measurements.
  > **Utility:** Provides empirical, real-world baseline distributions for $N, P, K,$ and $\text{pH}$ across Kerala's 14 districts, showing that >68% of Kerala's cultivated soils are naturally acidic ($\text{pH } 4.5–5.8$) and low in available potassium.

---

### 2.5 Kerala Agricultural University (KAU) & ICAR Specialized Research Institutes
Definitive scientific authority for crop environmental envelopes, nutrient responses, and tolerance thresholds:

#### A. Kerala Agricultural University (KAU) — Directorate of Extension, Mannuthy
* **Core Reference:** *Package of Practices Recommendations: Crops* (15th–17th Editions). [https://kau.in](https://kau.in)
* **Agronomic Envelopes Documented:**
  - Standard agronomic requirements ($N, P, K$ fertilizer doses in $\text{kg/ha}$, optimal soil $\text{pH}$, annual rainfall ranges, temperature limits, elevation tolerances, and drainage requirements) for:
    - Cereals (Rice — Virppu, Mundakan, Puncha; upland rice)
    - Plantation Crops (Coconut, Rubber, Arecanut, Cashew, Cocoa)
    - Tuber Crops (Tapioca/Cassava, Sweet potato, Elephant foot yam, Colocasia, Yams)
    - Spices (Black Pepper, Cardamom, Ginger, Turmeric, Clove, Nutmeg, Cinnamon)
    - Fruits (Banana — Nendran, Robusta, Poovan; Pineapple, Mango, Papaya)
    - Vegetables (Tomato, Chilli, Brinjal, Okra, Bitter gourd, Snake gourd, Amaranthus)

#### B. ICAR-CPCRI (Central Plantation Crops Research Institute, Kudlu, Kasaragod)
* **Portal:** [https://cpcri.icar.gov.in](https://cpcri.icar.gov.in)
* **Coverage:** Coconut, Arecanut, Cocoa, and Cashew in humid tropical and coastal agro-ecosystems.
* **Key Data:** Exact nutrient removal rates per palm/tree per year, root-zone depth, drought sensitivity, and response to soil acidity amelioration (dolomite application).

#### C. ICAR-CTCRI (Central Tuber Crops Research Institute, Sreekariyam, Thiruvananthapuram)
* **Portal:** [https://ctcri.org](https://ctcri.org)
* **Coverage:** Tapioca (Cassava), Elephant foot yam (*Amorphophallus paeoniifolius*), Sweet potato (*Ipomoea batatas*), and Chinese potato.
* **Key Data:** Starch bulking nutrient ratios (high potassium requirement: $N:P_2O_5:K_2O = 100:50:100 \text{ kg/ha}$), drought hardiness, soil drainage constraints, and laterite soil suitability.

#### D. ICAR-IISR (Indian Institute of Spices Research, Chelavoor, Kozhikode)
* **Portal:** [https://spices.res.in](https://spices.res.in)
* **Coverage:** Black Pepper, Cardamom, Ginger, Turmeric, Nutmeg, Clove, Cinnamon.
* **Key Data:** High-range Western Ghats agro-ecological requirements, shade tree canopy dynamics, elevation envelopes (Cardamom strictly $>800\text{ m}$ MSL; Pepper up to $1,200\text{ m}$ MSL), susceptibility to waterlogging and *Phytophthora* foot rot.

#### E. Rubber Board of India & RRII (Rubber Research Institute of India, Kottayam)
* **Portal:** [https://rubberboard.gov.in](https://rubberboard.gov.in)
* **Coverage:** Natural Rubber (*Hevea brasiliensis*) cultivation across Kerala and Kanyakumari.
* **Key Data:** Acidic laterite soil tolerance ($\text{pH } 4.5–6.0$), high rainfall requirement ($2,000–3,500\text{ mm}$), temperature threshold ($20–35^\circ\text{C}$), tapping panel management, and clone-specific nutrient responses (e.g., RRII 105, RRII 400 series).

---

### 2.6 Meteorological Repositories (IMD & NASA POWER)
* **Source Organizations:** India Meteorological Department (IMD Pune/New Delhi) & NASA Prediction of Worldwide Energy Resources (POWER) Agroclimatology.
* **Official Portals:** [https://mausam.imd.gov.in](https://mausam.imd.gov.in) | [https://custom-agro.imd.gov.in](https://custom-agro.imd.gov.in) | [https://power.larc.nasa.gov](https://power.larc.nasa.gov)
* **Geographic Coverage:** 14 District Agro-Meteorological Units (DAMUs) of Kerala, Automated Weather Stations (AWS), and $0.5^\circ \times 0.5^\circ$ gridded satellite-reanalysis points.
* **Available Columns & Metrics:**
  - `Daily & Monthly Rainfall` ($\text{mm}$)
  - `Maximum, Minimum, and Mean Temperature` ($^\circ\text{C}$)
  - `Morning & Evening Relative Humidity` (%)
  - `Solar Radiation` ($\text{MJ/m}^2/\text{day}$)
  - `Wind Speed & Reference Evapotranspiration` ($ET_0$)
* **Climatological Characteristics of Kerala:**
  - Average annual precipitation: $2,800–3,200\text{ mm}$ (South-West Monsoon June–Sept delivers ~70%, North-East Monsoon Oct–Nov delivers ~20%, pre-monsoon ~10%).
  - Palakkad gap / rain-shadow: $1,600–2,100\text{ mm}$.
  - High Ranges (Peermade, Munnar, Wayanad): $3,500–4,500+\text{ mm}$.
  - Mean temperature: Coastal/Midland $24–33^\circ\text{C}$; Highland $12–25^\circ\text{C}$.
  - Humidity: Persistently elevated ($70–95\%$).

---

### 2.7 Academic & Kaggle Datasets
* **Atharva Ingle Benchmark Dataset (2020):**
  - Columns: `N, P, K, temperature, humidity, ph, rainfall, label`
  - Samples: 2,200 rows (exactly 100 rows per crop across 22 classes)
  - Provenance: Synthesized based on agronomic centroids with Gaussian perturbations.
  - Role in project: Baseline control benchmark; preserved for regression testing and backwards compatibility.
* **ICRISAT District Level Database (DLD):**
  - Portal: [https://vdsa.icrisat.org](https://vdsa.icrisat.org)
  - Meso-level historical database (1966–2017) tracking 19 major states.
  - Limitation: Focuses on Semi-Arid Tropics (SAT); Kerala representation is sparse compared to central/northern states, and does not contain farm-level $N, P, K$.
* **Soil Nutrient Dataset of Southern Indian States (Kaggle / Manoj Kengalagutti):**
  - Columns: `State, District, N, P, K, pH`
  - Coverage: Southern states including Karnataka, Kerala, Tamil Nadu.
  - Limitation: Measures soil nutrients only; has no target crop label, no weather parameters, and no yield data.

---

## 3. Dataset Comparison Matrix

| Source Name | Primary Publisher | Geographic Granularity | Parameters Available | Target Crops Included | Micro vs. Macro Reality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EARAS Kerala** | DES Kerala | District / Taluk / Village cuts | Area, Production, Yield, Land Use | Paddy, Tapioca, Coconut, Rubber, Arecanut, Pepper, Cardamom, Cashew, Ginger | **Macro Aggregate:** Contains no soil chemistry or microclimate measurements. |
| **GoI APY / UPAg** | Min. of Agriculture | District / Season | Area, Production, Yield | All major field & plantation crops across India | **Macro Aggregate:** Official yield figures, zero soil test parameters. |
| **Soil Health Card** | DA&FW (GoI) | Village / District / State | N, P, K, pH, EC, OC, Micronutrients | Unlinked to continuous yield / target crop | **Micro Soil Census:** Real soil tests, but uncoupled from weather and crop outcomes. |
| **KAU Package of Practices** | Kerala Agricultural Univ. | Agro-Ecological Units (AEU 1–23) | Agronomic N-P-K doses, pH ranges, Rain, Temp, Soil type | Complete Kerala crop inventory (Plantations, Spices, Tubers, Vegs) | **Agronomic Ground Truth:** Scientific requirement envelopes and field trial baselines. |
| **IMD Climatology** | India Meteorological Dept. | District AWS / Gridded | Rain, Max/Min Temp, Humidity, $ET_0$ | Climate only (independent of crops) | **Micro/Meso Climate:** Continuous real-world meteorological observations. |
| **Original Kaggle Benchmark** | Atharva Ingle | Abstract / Universal | N, P, K, Temp, Humidity, pH, Rain, Label | 22 classes (only 4 Kerala-relevant: Rice, Coconut, Banana, Coffee) | **Synthetic Point Data:** Idealized clusters; lacks Kerala commercial staples. |

---

## 4. Scientific Conclusion & Path Forward

1. **No Single "Magic" Real-World CSV Exists:** There is no single open-access government file in India that simultaneously records farmer-level $N, P, K, \text{pH}$, sensor-measured temperature/humidity/rainfall, and verified crop yield across 40+ crops in a single table.
2. **Rejection of Unlabeled Fabrication:** Past informal projects have simply taken random numbers within broad bounds and merged them into a fake dataset. We strictly reject this approach. Any expanded dataset must explicitly record `data_source` for every row.
3. **Multi-Source Hybrid Strategy:**
   - The original 2,200 benchmark records are preserved and explicitly tagged with `data_source = "Kaggle Benchmark (Semi-Synthetic)"`.
   - Verified Kerala agricultural records are curated using empirical Soil Health Card district distributions, KAU/ICAR Package of Practices agronomic envelopes, IMD district meteorological normals, and EARAS district yield baselines.
   - Every observation specifies `district`, `season`, `soil_type`, `irrigation`, `yield`, and `data_source`.
   - The ML evaluation explicitly compares model performance across benchmark, expanded, and regional holdout splits.
