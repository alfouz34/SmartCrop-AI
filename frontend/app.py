import os
import sys
from pathlib import Path
import requests
import streamlit as st
import plotly.express as px
import pandas as pd

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import recommend_crops, load_model

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Check backend availability or initialize standalone model
api_available = False
local_model = None
model_load_error = None

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.0)
    if health_resp.status_code == 200:
        api_available = True
except Exception:
    api_available = False

if not api_available:
    try:
        local_model = load_model()
    except Exception as exc:
        local_model = None
        model_load_error = str(exc)

st.set_page_config(
    page_title="SmartCrop AI",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 SmartCrop AI")
st.subheader("Intelligent Crop Recommendation System")
st.caption(
    "Decision support tool utilizing machine learning to recommend optimal crops based on soil nutrients and regional climate conditions."
)

# API / Standalone Health Check in Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    if api_available:
        st.success("API Connected: Online")
    elif local_model is not None:
        st.success("Standalone Mode — Model Loaded")
    else:
        st.error(f"Model Load Error: {model_load_error or 'Could not load local model'}")

    st.header("🌾 Agricultural Inputs")

    # Quick test presets
    preset = st.selectbox(
        "Load Example Profile",
        ["Custom", "Rice (Wetland / High Rainfall)", "Apple (High P & K Orchard)", "Chickpea (Arid / Low Humidity)"],
    )

    defaults = {
        "Custom": (90.0, 42.0, 43.0, 25.0, 80.0, 6.5, 100.0),
        "Rice (Wetland / High Rainfall)": (80.0, 48.0, 40.0, 24.0, 82.0, 6.4, 235.0),
        "Apple (High P & K Orchard)": (20.0, 134.0, 200.0, 22.0, 92.0, 6.0, 112.0),
        "Chickpea (Arid / Low Humidity)": (40.0, 68.0, 80.0, 19.0, 17.0, 7.3, 80.0),
    }
    d_n, d_p, d_k, d_temp, d_hum, d_ph, d_rain = defaults[preset]

    st.subheader("Soil Nutrients (kg/ha)")
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=d_n, step=1.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=d_p, step=1.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=400.0, value=d_k, step=1.0)

    st.subheader("Climate & Soil Chemistry")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=d_temp, step=0.5)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=d_hum, step=1.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=d_ph, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=d_rain, step=1.0)

    submitted = st.button("Recommend Crops", type="primary", use_container_width=True)

col_info, col_summary = st.columns([1.2, 1])
with col_info:
    st.markdown("### 🔍 How It Works")
    st.write(
        "SmartCrop AI evaluates your 7 soil and climate inputs against 22 distinct crop profiles "
        "using a trained Random Forest model (99.3%+ validation F1-score). It returns the top 3 best-matched "
        "crops ranked by model prediction scores."
    )
    st.info(
        "💡 **Note on Probabilities**: The Random Forest's `predict_proba()` function outputs model-estimated class "
        "probabilities by averaging the class probability estimates produced by the individual decision trees, which have "
        "not been calibrated. Treat these values as relative model prediction scores rather than guaranteed real-world success rates."
    )

with col_summary:
    st.markdown("### 📋 Current Input Summary")
    summary_df = pd.DataFrame({
        "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"],
        "Value": [f"{n:.0f} kg/ha", f"{p:.0f} kg/ha", f"{k:.0f} kg/ha", f"{temperature:.1f} °C", f"{humidity:.0f} %", f"{ph:.2f}", f"{rainfall:.1f} mm"],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

if submitted:
    payload = {
        "N": n, "P": p, "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    with st.spinner("Analyzing soil and climate data..."):
        recs = None
        advisory_note = (
            "Model scores represent statistical algorithm outputs and not guaranteed real-world yield or success. "
            "Always verify choices with regional agronomists and current market conditions."
        )

        # 1. Try FastAPI backend if online
        if api_available:
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    recs = data.get("recommendations", [])
                    advisory_note = data.get("note", advisory_note)
            except Exception:
                recs = None

        # 2. Standalone fallback using local production model
        if recs is None:
            try:
                recs = recommend_crops(payload, model=local_model, top_n=3)
            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")

        if recs:
            st.divider()
            st.markdown("## 🌾 Top 3 Crop Recommendations")

            cards = st.columns(len(recs))
            medals = ["🥇 Rank 1", "🥈 Rank 2", "🥉 Rank 3"]
            for i, (card, item) in enumerate(zip(cards, recs)):
                with card:
                    st.subheader(medals[i] if i < len(medals) else f"Rank {i+1}")
                    st.metric(label="Recommended Crop", value=item["crop"].title())
                    if item["score"] is not None:
                        st.metric(label="Model Prediction Score", value=f"{item['score'] * 100:.1f}%")

            if all(item["score"] is not None for item in recs):
                chart_df = pd.DataFrame(recs)
                chart_df["crop_name"] = chart_df["crop"].str.title()
                chart_df["score_pct"] = chart_df["score"] * 100
                fig = px.bar(
                    chart_df,
                    x="crop_name",
                    y="score_pct",
                    color="score_pct",
                    color_continuous_scale="Greens",
                    text=chart_df["score_pct"].apply(lambda v: f"{v:.1f}%"),
                    title="Top 3 Crop Prediction Scores",
                    labels={"crop_name": "Crop", "score_pct": "Model Prediction Score (%)"},
                    range_y=[0, 105],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.warning("⚠️ **Agronomic Advisory**: " + advisory_note)

st.divider()
st.header("📊 Understanding the Model")

col_exp_chart, col_exp_text = st.columns([1.2, 1])

with col_exp_chart:
    st.subheader("Global Feature Importance (Random Forest)")
    # Actual calculated Gini importance values
    importance_data = pd.DataFrame({
        "Feature": [
            "Soil pH",
            "Temperature",
            "Nitrogen (N)",
            "Phosphorus (P)",
            "Potassium (K)",
            "Relative Humidity",
            "Rainfall",
        ],
        "Importance (%)": [5.39, 7.58, 10.39, 15.37, 17.92, 21.19, 22.15],
    })
    fig_imp = px.bar(
        importance_data,
        x="Importance (%)",
        y="Feature",
        orientation="h",
        text=importance_data["Importance (%)"].apply(lambda v: f"{v:.1f}%"),
        color="Importance (%)",
        color_continuous_scale="Greens",
        range_x=[0, 26],
    )
    fig_imp.update_layout(
        showlegend=False,
        margin=dict(l=10, r=20, t=20, b=20),
        height=340,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col_exp_text:
    st.subheader("What the 7 Input Features Mean")
    st.markdown(
        """
        - **Rainfall (22.2%)**: Primary differentiator separating wetland crops (rice, jute) from arid crops.
        - **Humidity (21.2%)**: Crucial in distinguishing low-humidity pulses (chickpea) from tropical fruits.
        - **Potassium / K (17.9%) & Phosphorus / P (15.4%)**: Distinctive requirements for specialized fruits (apple, grapes).
        - **Nitrogen / N (10.4%)**: Separates heavy commercial feeders (cotton, coffee) from nitrogen-fixing legumes.
        - **Temperature (7.6%) & Soil pH (5.4%)**: Provide baseline agricultural feasibility, though many crops overlap in standard pH (5.5–7.5).
        """
    )

st.info(
    "📌 **Global vs. Local Explanation**: This chart represents **global feature importance**, showing the overall "
    "behaviour of the trained Random Forest model. The model was trained on 1,760 records, with the remaining 440 records "
    "reserved as a held-out test set. It describes general model behaviour across training splits, not why a specific individual "
    "prediction was chosen. A feature with lower global importance (like pH) can still be decisive for individual edge-case crops."
)

st.divider()
st.caption("SmartCrop AI Decision Support System • Powered by FastAPI & Streamlit")
