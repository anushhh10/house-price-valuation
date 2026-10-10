"""
Multi-Agent Explainable House Price Prediction SaaS Platform (INR - ₹).

Enterprise-grade residential valuation platform integrating CatBoost gradient boosting,
SHAP local explainability, and specialized autonomous domain agents
(Infrastructure, Safety, Environment, Growth) tailored to buyer personas.
"""

import os
import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

from agents import (
    MultiAgentOrchestrator,
    VALID_BUYER_PROFILES,
    PROFILE_DIMENSION_WEIGHTS,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Explainable House Price Valuation (Chennai)",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Currency Formatter Utility Function
def format_inr(amount: float, show_sign: bool = False, decimals: int = 2) -> str:
    """
    Intelligently format an amount in Indian Rupees (₹) to Lakhs (L) or Crores (Cr).
    Guarantees clean sign presentation without double minus signs.
    """
    abs_val = abs(amount)
    sign = ""
    if show_sign:
        sign = "+" if amount > 0 else ("-" if amount < 0 else "")
    elif amount < 0:
        sign = "-"

    if abs_val >= 10_000_000:
        cr_val = abs_val / 10_000_000
        return f"{sign}₹{cr_val:.{decimals}f} Cr"
    elif abs_val >= 100_000:
        l_val = abs_val / 100_000
        return f"{sign}₹{l_val:.{decimals}f} L"
    else:
        return f"{sign}₹{abs_val:,.0f}"


# Custom High-End SaaS Glassmorphism Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Top Brand Navigation Bar */
    .brand-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(19, 24, 34, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(14px);
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #38BDF8 0%, #6366F1 50%, #EC4899 100%);
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .brand-tagline {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 1px;
    }

    /* Action Panel */
    .action-panel {
        background: #1E232A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: #1E232A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
    }

    /* KPI Row Cards */
    .kpi-card {
        background: #1E232A;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 15px;
        padding: 18px 20px;
        height: 154px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }
    .kpi-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #94A3B8;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value {
        font-size: 1.88rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.15;
        margin: 4px 0;
    }

    /* Status Pill Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        width: fit-content;
    }
    .status-pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-pill-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .status-pill-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    .status-pill-yellow {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .status-pill-red {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    /* Agent Cards */
    .agent-box {
        background: #1E232A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .agent-box:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .agent-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .insight-bullet {
        font-size: 0.82rem;
        color: #CBD5E1;
        line-height: 1.45;
        margin-bottom: 5px;
    }

    /* Sub-metric progress container */
    .submetric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 2px;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        padding-bottom: 4px;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px 10px 0 0;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94A3B8;
        border: none;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* 5-Tier Architecture Step Cards */
    .tier-box {
        background: #1E232A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.24);
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .tier-box:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-3px);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.38);
    }
    .tier-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        padding: 2px 7px;
        border-radius: 5px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .tier-heading {
        font-size: 0.92rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 3px;
    }
    .tier-subhead {
        font-size: 0.76rem;
        font-weight: 600;
        color: #818CF8;
        margin-bottom: 8px;
    }
    .tier-text {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model & Explainer Loading Functions (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    """Load pre-trained model, SHAP explainer, feature list, and metrics."""
    model_path = "models/best_model.joblib"
    explainer_path = "models/shap_explainer.joblib"
    features_path = "models/feature_names.json"
    metrics_path = "models/metrics.json"

    if not os.path.exists(model_path) or not os.path.exists(explainer_path):
        return None, None, None, None

    model = joblib.load(model_path)
    explainer = joblib.load(explainer_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return model, explainer, feature_names, metrics


model, explainer, feature_names, metrics = load_ml_pipeline()
orchestrator = MultiAgentOrchestrator()


# -----------------------------------------------------------------------------
# Chennai Locality Presets & Societal Metric Baselines
# -----------------------------------------------------------------------------
LOCALITY_PRESETS = {
    "Anna Nagar (Residential & Educational Hub)": {
        "transit_score": 8.5,
        "school_score": 9.2,
        "hospital_dist": 1.5,
        "crime_rate": 15.0,
        "police_dist": 1.2,
        "aqi": 48,
        "green_cover": 40.0,
        "noise_level": 46.0,
        "growth_index": 78.0,
        "hub_dist": 3.5,
    },
    "OMR - Thoraipakkam / Sholinganallur (IT Corridor)": {
        "transit_score": 8.0,
        "school_score": 7.8,
        "hospital_dist": 2.8,
        "crime_rate": 22.0,
        "police_dist": 2.4,
        "aqi": 58,
        "green_cover": 32.0,
        "noise_level": 52.0,
        "growth_index": 92.0,
        "hub_dist": 1.2,
    },
    "Velachery (Transit & Multi-Hub Commercial)": {
        "transit_score": 9.0,
        "school_score": 8.0,
        "hospital_dist": 2.0,
        "crime_rate": 25.0,
        "police_dist": 1.8,
        "aqi": 65,
        "green_cover": 26.0,
        "noise_level": 56.0,
        "growth_index": 85.0,
        "hub_dist": 2.2,
    },
    "Adyar / Besant Nagar (Coastal & High Livability)": {
        "transit_score": 7.5,
        "school_score": 9.0,
        "hospital_dist": 1.6,
        "crime_rate": 18.0,
        "police_dist": 1.4,
        "aqi": 38,
        "green_cover": 46.0,
        "noise_level": 42.0,
        "growth_index": 72.0,
        "hub_dist": 3.8,
    },
    "T. Nagar (Commercial Core)": {
        "transit_score": 9.5,
        "school_score": 8.2,
        "hospital_dist": 1.2,
        "crime_rate": 32.0,
        "police_dist": 1.0,
        "aqi": 82,
        "green_cover": 18.0,
        "noise_level": 65.0,
        "growth_index": 88.0,
        "hub_dist": 0.8,
    },
}

LOCALITY_OPTIONS = [
    "Anna Nagar (Residential & Educational Hub)",
    "OMR - Thoraipakkam / Sholinganallur (IT Corridor)",
    "Velachery (Transit & Multi-Hub Commercial)",
    "Adyar / Besant Nagar (Coastal & High Livability)",
    "T. Nagar (Commercial Core)",
    "Custom / Manual Input",
]

# Initialize Session State defaults with Anna Nagar
if "initialized_locality_presets" not in st.session_state:
    st.session_state["initialized_locality_presets"] = True
    st.session_state["selected_locality_box"] = LOCALITY_OPTIONS[0]
    default_vals = LOCALITY_PRESETS[LOCALITY_OPTIONS[0]]
    for key, val in default_vals.items():
        st.session_state[key] = val


def on_locality_change():
    chosen = st.session_state.selected_locality_box
    if chosen in LOCALITY_PRESETS:
        presets = LOCALITY_PRESETS[chosen]
        for key, val in presets.items():
            st.session_state[key] = val


# -----------------------------------------------------------------------------
# Valuation Certificate Generator Utility
# -----------------------------------------------------------------------------
def generate_valuation_report(
    property_data: dict,
    buyer_profile: str,
    locality: str,
    raw_price: float,
    multiplier: float,
    adjusted_price: float,
    agent_report: dict,
    shap_top_pos: list,
    shap_top_neg: list,
) -> str:
    """Generate a clean, structured Valuation Summary Certificate in text format."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    report_id = f"VAL-CHN-{int(time.time())}"

    lines = [
        "=" * 74,
        "   MULTI-AGENT EXPLAINABLE HOUSE PRICE VALUATION (CHENNAI) CERTIFICATE   ",
        "=" * 74,
        f"Certificate ID   : {report_id}",
        f"Timestamp        : {now_str}",
        f"Chennai Locality : {locality}",
        f"Active Persona   : {buyer_profile}",
        f"Valuation Unit   : Indian Rupees (INR - ₹)",
        "-" * 74,
        "",
        "1. EXECUTIVE VALUATION SUMMARY",
        f"   - Target Chennai Locality            : {locality}",
        f"   - Baseline Machine Learning Estimate : {format_inr(raw_price)} (₹{raw_price:,.0f})",
        f"   - Persona-Weighted Multiplier        : {multiplier:.3f}x",
        f"   - Final Persona-Adjusted Valuation   : {format_inr(adjusted_price)} (₹{adjusted_price:,.0f})",
        f"   - Locational Livability Index        : {agent_report['composite_score']}/100",
        "",
        "2. PROPERTY STRUCTURAL SPECIFICATIONS",
        f"   - Carpet Area      : {property_data['sqft']} sqft",
        f"   - Configuration    : {property_data['bhk']} BHK, {property_data['bathrooms']} Bathrooms",
        f"   - Property Age     : {property_data['property_age']} Years",
        f"   - Floor Level      : Floor {property_data['floor_number']} of {property_data['total_floors']}",
        f"   - Parking Capacity : {property_data['parking_spaces']} Dedicated Bays",
        "",
        "3. AUTONOMOUS DOMAIN AGENT AUDIT",
    ]

    for agent_key, title in [
        ("InfrastructureAgent", "Infrastructure Agent"),
        ("SafetyAgent", "Safety Agent"),
        ("EnvironmentAgent", "Environment Agent"),
        ("GrowthAgent", "Growth Agent"),
    ]:
        ev = agent_report["agent_evaluations"][agent_key]
        lines.append(f"   [{title}]")
        lines.append(f"     Score: {ev.dimension_score}/100 | Category: {ev.rating_category} | Price Impact: {ev.price_impact_factor:.3f}x")
        for ins in ev.insights:
            lines.append(f"     • {ins}")
        lines.append("")

    lines.extend([
        "4. EXPLAINABLE AI (SHAP) ATTRIBUTION HIGHLIGHTS",
        "   Top Positive Value Boosters:",
    ])
    for item in shap_top_pos:
        lines.append(f"     + {item['feature']} ({item['val']}): {item['impact']}")

    lines.append("   Top Value Penalizers:")
    for item in shap_top_neg:
        lines.append(f"     - {item['feature']} ({item['val']}): {item['impact']}")

    lines.extend([
        "",
        "-" * 74,
        "System Engine: CatBoost Regressor v1.2.10 (R² = 0.9809, Test MAE = ₹3.53 Lakhs)",
        "Certified by Multi-Agent Explainable AI Valuation Framework (Chennai Edition).",
        "=" * 74,
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Sidebar: Interactive Controls & Locality / Persona Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📍 Chennai Locality Focus")

    selected_locality = st.selectbox(
        "📍 Select Chennai Locality",
        options=LOCALITY_OPTIONS,
        key="selected_locality_box",
        on_change=on_locality_change,
        help="Choose a pre-calibrated Chennai locality to load real-world baseline societal metrics, or select Custom / Manual Input.",
    )

    if selected_locality != "Custom / Manual Input":
        st.caption(f"✨ Preset baseline loaded for **{selected_locality.split(' (')[0]}**. Sliders can still be fine-tuned below.")
    else:
        st.caption("🛠️ Custom mode active: Adjust societal sliders freely.")

    st.markdown("---")
    st.markdown("### 👤 Buyer Persona Profile")

    persona_pill_options = [
        "👨‍👩‍👧 Family",
        "💼 Investor",
        "🎓 Student",
        "👨‍💻 Professional",
    ]

    selected_persona_pill = st.pills(
        "Select Active Persona",
        options=persona_pill_options,
        default="👨‍👩‍👧 Family",
        label_visibility="collapsed",
    )

    persona_code_map = {
        "👨‍👩‍👧 Family": "Family",
        "💼 Investor": "Investor",
        "🎓 Student": "Student",
        "👨‍💻 Professional": "Working Professional",
    }
    buyer_profile = persona_code_map.get(selected_persona_pill, "Family")

    persona_descriptions = {
        "Family": "👨‍👩‍👧 **Family**: Focuses on school district excellence, neighborhood security, clean air, and peaceful surroundings.",
        "Investor": "💼 **Investor**: Prioritizes capital appreciation momentum, commercial corridors, and high-yield growth velocity.",
        "Student": "🎓 **Student**: Prioritizes mass transit proximity, budget efficiency, and pedestrian safety.",
        "Working Professional": "👨‍💻 **Professional**: Focuses on rapid commercial hub commute, transit speed, and low ambient noise for remote work.",
    }
    st.info(persona_descriptions[buyer_profile])

    st.markdown("---")
    st.markdown("### 🏢 Structural Specifications")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sqft = st.number_input("Sq. Footage (sqft)", min_value=400, max_value=5000, value=1550, step=50)
        bhk = st.slider("BHK", min_value=1, max_value=5, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
    with col_s2:
        property_age = st.slider("Age (Years)", min_value=0, max_value=40, value=4)
        floor_number = st.slider("Floor", min_value=1, max_value=30, value=6)
        total_floors = st.slider("Total Floors", min_value=2, max_value=35, value=16)
        parking_spaces = st.slider("Parking", min_value=0, max_value=3, value=1)

    st.markdown("---")
    st.markdown("### 📍 Localized Societal Factors")

    with st.expander("🚆 Infrastructure & Civic Amenities", expanded=True):
        transit_score = st.slider("Transit Access Score (1-10)", min_value=1.0, max_value=10.0, step=0.1, key="transit_score")
        school_score = st.slider("School District Quality (1-10)", min_value=1.0, max_value=10.0, step=0.1, key="school_score")
        hospital_dist = st.slider("Distance to Hospital (km)", min_value=0.2, max_value=15.0, step=0.1, key="hospital_dist")

    with st.expander("🛡 Public Safety & Security", expanded=True):
        crime_rate = st.slider("Crime Rate Index (Lower is safer)", min_value=5.0, max_value=95.0, step=1.0, key="crime_rate")
        police_dist = st.slider("Distance to Police Station (km)", min_value=0.2, max_value=12.0, step=0.1, key="police_dist")

    with st.expander("🌿 Environment & Acoustic Wellness", expanded=False):
        aqi = st.slider("Air Quality Index (AQI)", min_value=25, max_value=280, step=1, key="aqi")
        green_cover = st.slider("Green Canopy Cover (%)", min_value=5.0, max_value=65.0, step=1.0, key="green_cover")
        noise_level = st.slider("Ambient Noise Level (dB)", min_value=35.0, max_value=85.0, step=1.0, key="noise_level")

    with st.expander("📈 Economic Velocity & Growth", expanded=False):
        growth_index = st.slider("Neighborhood Growth Index (1-100)", min_value=5.0, max_value=98.0, step=1.0, key="growth_index")
        hub_dist = st.slider("Distance to Commercial Hub (km)", min_value=0.5, max_value=25.0, step=0.1, key="hub_dist")


# Assemble Input Dictionary & DataFrame
property_input = {
    "sqft": sqft,
    "bhk": bhk,
    "bathrooms": bathrooms,
    "property_age": property_age,
    "floor_number": floor_number,
    "total_floors": total_floors,
    "parking_spaces": parking_spaces,
    "school_score": school_score,
    "hospital_distance_km": hospital_dist,
    "transit_score": transit_score,
    "crime_rate": crime_rate,
    "police_proximity_km": police_dist,
    "aqi": aqi,
    "green_cover_pct": green_cover,
    "noise_level_db": noise_level,
    "growth_index": growth_index,
    "commercial_hub_distance_km": hub_dist,
}

input_df = pd.DataFrame([property_input])[feature_names] if feature_names else pd.DataFrame([property_input])


if model is None:
    st.error("⚠️ Pre-trained model artifacts not found in `models/`. Please execute `python models/train_model.py` first.")
    st.stop()


# -----------------------------------------------------------------------------
# Evaluation: ML Prediction + Multi-Agent Orchestration
# -----------------------------------------------------------------------------
raw_ml_price = float(model.predict(input_df)[0])
agent_report = orchestrator.evaluate_property(property_input, buyer_profile=buyer_profile)
multiplier = agent_report["composite_multiplier"]
adjusted_price = round(raw_ml_price * multiplier, -3)
composite_score = agent_report["composite_score"]
dollar_diff = adjusted_price - raw_ml_price
pct_diff = ((multiplier - 1.0) * 100.0)

# Extract SHAP top drivers for report generation & UI display
shap_values_obj = explainer(input_df)
base_expected_val = float(shap_values_obj.base_values[0])
pred_val = float(shap_values_obj.values[0].sum() + base_expected_val)

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Value": [property_input[f] for f in feature_names],
    "Impact_Raw": shap_values_obj.values[0],
})
contrib_df["AbsImpact"] = contrib_df["Impact_Raw"].abs()
contrib_df = contrib_df.sort_values(by="AbsImpact", ascending=False)

top_pos_df = contrib_df[contrib_df["Impact_Raw"] > 0].head(4)
top_neg_df = contrib_df[contrib_df["Impact_Raw"] < 0].head(4)

shap_pos_list = [
    {"feature": r["Feature"], "val": r["Value"], "impact": format_inr(r["Impact_Raw"], show_sign=True)}
    for _, r in top_pos_df.iterrows()
]
shap_neg_list = [
    {"feature": r["Feature"], "val": r["Value"], "impact": format_inr(r["Impact_Raw"], show_sign=True)}
    for _, r in top_neg_df.iterrows()
]


# -----------------------------------------------------------------------------
# Top Brand Bar / Main Header Banner (With Prominent Chennai Locality)
# -----------------------------------------------------------------------------
locality_short = selected_locality.split(" (")[0]

st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="brand-logo">🏢</div>
            <div>
                <h1 class="brand-title">Multi-Agent Explainable House Price Valuation (Chennai)</h1>
                <div class="brand-tagline">
                    📍 Target Locality: <b style="color: #38BDF8; font-weight: 700;">{selected_locality}</b>
                    &nbsp;|&nbsp; Autonomous Multi-Agent Reasoning & SHAP Explainability (INR - ₹)
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span class="status-pill status-pill-blue" style="font-size: 0.82rem; padding: 5px 12px;">📍 {locality_short}</span>
            <span class="status-pill status-pill-green">🟢 CatBoost Engine Active</span>
            <span class="status-pill status-pill-purple">⚡ SHAP Interpretability Online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Action Trigger & Report Export Bar
# -----------------------------------------------------------------------------
act_col1, act_col2, act_col3 = st.columns([5, 3, 3])

with act_col1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; height: 100%; flex-wrap: wrap;">
            <span style="font-size: 0.85rem; color: #94A3B8;">Locality:</span>
            <span class="status-pill status-pill-blue"><b>📍 {locality_short}</b></span>
            <span style="font-size: 0.85rem; color: #94A3B8;">Persona:</span>
            <span class="status-pill status-pill-purple"><b>{selected_persona_pill}</b></span>
            <span style="font-size: 0.82rem; color: #64748B;">• {sqft} sqft • {bhk} BHK • Age {property_age}y</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with act_col2:
    if st.button("⚡ Run Instant Valuation", type="primary", use_container_width=True):
        st.toast(f"⚡ Valuation re-computed for {locality_short} under {buyer_profile} persona!", icon="✅")

with act_col3:
    safe_loc = locality_short.replace(" ", "_").replace(".", "").replace("/", "-").lower()
    valuation_report_text = generate_valuation_report(
        property_input,
        buyer_profile,
        selected_locality,
        raw_ml_price,
        multiplier,
        adjusted_price,
        agent_report,
        shap_pos_list,
        shap_neg_list,
    )
    st.download_button(
        label="📄 Export Valuation Audit (TXT)",
        data=valuation_report_text,
        file_name=f"valuation_audit_chennai_{safe_loc}_{buyer_profile.lower()}_{int(time.time())}.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top KPI Banner: Perfectly Aligned 4-Column Metric Row
# -----------------------------------------------------------------------------
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">🤖 Baseline ML Estimate</div>
                <div class="kpi-value">{format_inr(raw_ml_price)}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="status-pill status-pill-blue">{metrics['best_model']} Model</span>
                <span style="font-size: 0.72rem; color: #94A3B8;">R² {metrics['metrics'][metrics['best_model']]['R2']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    mult_pill_class = "status-pill-green" if multiplier >= 1.0 else "status-pill-red"
    mult_val_color = "#34D399" if multiplier >= 1.0 else "#F87171"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">📊 {buyer_profile} Multiplier</div>
                <div class="kpi-value" style="color:{mult_val_color};">{multiplier:.3f}x</div>
            </div>
            <span class="status-pill {mult_pill_class}">
                {format_inr(dollar_diff, show_sign=True)} ({pct_diff:+.1f}%)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-color: rgba(56, 189, 248, 0.45); background: #1B263B;">
            <div>
                <div class="kpi-title" style="color: #93C5FD;">💎 Final Persona Price</div>
                <div class="kpi-value" style="color: #60A5FA;">{format_inr(adjusted_price)}</div>
            </div>
            <span class="status-pill status-pill-blue">
                Calibrated for {buyer_profile}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    if composite_score >= 75:
        zone_badge = '<span class="status-pill status-pill-green">🟢 High Livability Zone</span>'
        score_color = "#34D399"
    elif composite_score >= 55:
        zone_badge = '<span class="status-pill status-pill-yellow">🟡 Moderate Livability Zone</span>'
        score_color = "#FBBF24"
    else:
        zone_badge = '<span class="status-pill status-pill-red">🔴 Developing Zone</span>'
        score_color = "#F87171"

    st.markdown(
        f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-title">⭐ Composite Livability Index</div>
                <div class="kpi-value" style="color:{score_color};">{composite_score:.1f}<span style="font-size:1.05rem; color:#94A3B8;">/100</span></div>
            </div>
            {zone_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Tabs: Agents Hub, SHAP Explainability, Benchmark
# -----------------------------------------------------------------------------
tab_agents, tab_shap, tab_benchmark = st.tabs(
    ["🤖 Multi-Agent Intelligence Hub", "🔍 SHAP Local Explainability", "📊 Model Performance & Architecture"]
)

category_pill_map = {
    "Excellent": "status-pill-green",
    "Good": "status-pill-blue",
    "Moderate": "status-pill-yellow",
    "Fair": "status-pill-yellow",
    "Poor": "status-pill-red",
}


# --- TAB 1: MULTI-AGENT INTELLIGENCE HUB ---
with tab_agents:
    st.markdown("#### 🧭 Autonomous Domain Agent Evaluations")
    col_chart, col_weights = st.columns([3, 2])

    with col_chart:
        categories = list(agent_report["radar_metrics"].keys())
        scores = list(agent_report["radar_metrics"].values())
        weights = [agent_report["dimension_weights"][c] * 100 for c in categories]

        categories_closed = categories + [categories[0]]
        scores_closed = scores + [scores[0]]
        weights_closed = weights + [weights[0]]

        fig_radar = go.Figure()

        fig_radar.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=categories_closed,
                fill="toself",
                name="Domain Score",
                line=dict(color="#38BDF8", width=2.5),
                fillcolor="rgba(56, 189, 248, 0.22)",
            )
        )

        fig_radar.add_trace(
            go.Scatterpolar(
                r=weights_closed,
                theta=categories_closed,
                fill="toself",
                name=f"{buyer_profile} Priority (%)",
                line=dict(color="#F43F5E", width=1.5, dash="dash"),
                fillcolor="rgba(244, 63, 94, 0.10)",
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8", gridcolor="rgba(148, 163, 184, 0.18)"),
                angularaxis=dict(color="#F1F5F9", gridcolor="rgba(148, 163, 184, 0.18)"),
                bgcolor="#1E232A",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=35, r=35, t=25, b=25),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#CBD5E1")),
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col_weights:
        st.markdown(f"**Persona Priority Calibration: `{buyer_profile}`**")
        for dim, w in agent_report["dimension_weights"].items():
            st.write(f"**{dim} Weight** ({int(w * 100)}%)")
            st.progress(float(w))

        st.caption(
            "Each agent adjusts internal feature importances and multiplier sensitivity dynamically based on the active buyer profile."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Domain Agent Rationales & Interactive Inspection Panels")

    agent_cols = st.columns(4)
    agent_instances = [
        ("InfrastructureAgent", "Infrastructure", "🚆", "#38BDF8"),
        ("SafetyAgent", "Safety", "🛡️", "#34D399"),
        ("EnvironmentAgent", "Environment", "🌿", "#A78BFA"),
        ("GrowthAgent", "Growth", "📈", "#FBBF24"),
    ]

    for col, (agent_key, title, icon, accent) in zip(agent_cols, agent_instances):
        eval_item = agent_report["agent_evaluations"][agent_key]
        pill_cls = category_pill_map.get(eval_item.rating_category, "status-pill-yellow")
        mult_str = f"{eval_item.price_impact_factor:.3f}x"

        with col:
            st.markdown(
                f"""
                <div class="agent-box">
                    <div>
                        <div class="agent-header">
                            <span class="agent-name">{icon} {title}</span>
                            <span class="status-pill {pill_cls}">{eval_item.rating_category}</span>
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: {accent};">
                            {eval_item.dimension_score}<span style="font-size:0.9rem; color:#94A3B8;">/100</span>
                        </div>
                        <div style="font-size:0.82rem; color:#94A3B8; margin-top: 4px; margin-bottom: 8px;">
                            Price Multiplier: <b style="color:#F1F5F9;">{mult_str}</b>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Agent Inspection & Explanation Button
            with st.expander("🔍 View Agent Rationale", expanded=False):
                st.markdown("**Key Qualitative Insights:**")
                for insight in eval_item.insights:
                    st.markdown(f"<p class='insight-bullet'>• {insight}</p>", unsafe_allow_html=True)

                st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                st.markdown("**Sub-Metric Decomposition:**")

                if agent_key == "InfrastructureAgent":
                    st.write(f"Transit Accessibility: `{transit_score}/10`")
                    st.progress(float(transit_score / 10.0))
                    st.write(f"School District Score: `{school_score}/10`")
                    st.progress(float(school_score / 10.0))
                    h_norm = max(0.0, min(1.0, 1.0 - (hospital_dist / 15.0)))
                    st.write(f"Hospital Proximity: `{hospital_dist} km`")
                    st.progress(float(h_norm))

                elif agent_key == "SafetyAgent":
                    c_norm = max(0.0, min(1.0, (100.0 - crime_rate) / 100.0))
                    st.write(f"Crime Safety Index: `{100 - crime_rate:.1f}/100`")
                    st.progress(float(c_norm))
                    p_norm = max(0.0, min(1.0, 1.0 - (police_dist / 12.0)))
                    st.write(f"Police Station Proximity: `{police_dist} km`")
                    st.progress(float(p_norm))

                elif agent_key == "EnvironmentAgent":
                    aqi_norm = max(0.0, min(1.0, (300.0 - aqi) / 300.0))
                    st.write(f"Air Quality Cleanliness: `AQI {aqi}`")
                    st.progress(float(aqi_norm))
                    st.write(f"Green Canopy Cover: `{green_cover}%`")
                    st.progress(float(green_cover / 65.0))
                    n_norm = max(0.0, min(1.0, (85.0 - noise_level) / 50.0))
                    st.write(f"Acoustic Serenity: `{noise_level} dB`")
                    st.progress(float(n_norm))

                elif agent_key == "GrowthAgent":
                    st.write(f"Neighborhood Growth Index: `{growth_index}/100`")
                    st.progress(float(growth_index / 100.0))
                    hub_norm = max(0.0, min(1.0, 1.0 - (hub_dist / 25.0)))
                    st.write(f"Commercial Hub Access: `{hub_dist} km`")
                    st.progress(float(hub_norm))


# --- TAB 2: SHAP LOCAL EXPLAINABILITY ---
with tab_shap:
    st.markdown("#### 🔬 SHAP (SHapley Additive exPlanations) Local Attribution")
    st.write(
        "Quantifies the exact Rupee-force (₹) contribution of each structural feature and localized societal indicator "
        "relative to the expected dataset baseline price."
    )

    shap_view_mode = st.radio(
        "Select SHAP Visualization Mode:",
        options=["🌊 Waterfall Attribution Plot", "📊 Force Drivers Bar Chart"],
        horizontal=True,
    )

    shap_c1, shap_c2 = st.columns([3, 2])

    with shap_c1:
        if shap_view_mode == "🌊 Waterfall Attribution Plot":
            st.markdown("##### Feature Force Waterfall Plot")
            try:
                fig_waterfall, ax = plt.subplots(figsize=(9, 6.2))
                plt.style.use("dark_background")
                shap.plots.waterfall(shap_values_obj[0], max_display=12, show=False)
                plt.title("SHAP Feature Attribution Waterfall (₹)", fontsize=11, color="#E2E8F0", pad=12)
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
            except Exception:
                st.info("Displaying Interactive Force Drivers breakdown:")
                chart_df = contrib_df.head(10).copy()
                chart_df = chart_df.sort_values(by="Impact_Raw", ascending=True)
                colors = ["#34D399" if x > 0 else "#F87171" for x in chart_df["Impact_Raw"]]
                fig_bar = go.Figure(
                    go.Bar(
                        x=chart_df["Impact_Raw"],
                        y=chart_df["Feature"],
                        orientation="h",
                        marker=dict(color=colors),
                        text=[format_inr(x, show_sign=True) for x in chart_df["Impact_Raw"]],
                        textposition="auto",
                    )
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#1E232A",
                    margin=dict(l=40, r=40, t=20, b=30),
                    height=420,
                    xaxis=dict(title="SHAP Value Contribution (₹)", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
                    yaxis=dict(color="#F1F5F9"),
                )
                st.plotly_chart(fig_bar, width="stretch")
        else:
            st.markdown("##### Interactive Force Drivers Bar Chart (INR)")
            chart_df = contrib_df.head(10).copy()
            chart_df = chart_df.sort_values(by="Impact_Raw", ascending=True)

            colors = ["#34D399" if x > 0 else "#F87171" for x in chart_df["Impact_Raw"]]

            fig_bar = go.Figure(
                go.Bar(
                    x=chart_df["Impact_Raw"],
                    y=chart_df["Feature"],
                    orientation="h",
                    marker=dict(color=colors),
                    text=[format_inr(x, show_sign=True) for x in chart_df["Impact_Raw"]],
                    textposition="auto",
                )
            )

            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1E232A",
                margin=dict(l=40, r=40, t=20, b=30),
                height=420,
                xaxis=dict(title="SHAP Value Contribution (₹)", gridcolor="rgba(148, 163, 184, 0.15)", color="#94A3B8"),
                yaxis=dict(color="#F1F5F9"),
            )
            st.plotly_chart(fig_bar, width="stretch")

    with shap_c2:
        st.markdown("##### 📌 Key Driver Decomposition (INR)")
        st.write(f"**Baseline Dataset Expected Price:** `{format_inr(base_expected_val)}`")
        net_contrib = pred_val - base_expected_val
        st.write(f"**Net Local Feature Force:** `{format_inr(net_contrib, show_sign=True)}`")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🟢 Top Positive Price Boosters:**")
        for _, r in top_pos_df.iterrows():
            val_formatted = format_inr(r["Impact_Raw"], show_sign=True)
            st.write(f"- **{r['Feature']}** (`{r['Value']}`): **{val_formatted}**")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔴 Top Value Penalizers:**")
        for _, r in top_neg_df.iterrows():
            val_formatted = format_inr(r["Impact_Raw"], show_sign=True)
            st.write(f"- **{r['Feature']}** (`{r['Value']}`): **{val_formatted}**")


# --- TAB 3: MODEL PERFORMANCE & ARCHITECTURE ---
with tab_benchmark:
    st.markdown("#### 🏆 Regression Model Leaderboard (Test Partition)")

    st.write(
        "Candidate regression models evaluated on a held-out 20% test partition (1,000 properties), "
        "denominated in Indian Rupees (INR - ₹)."
    )

    metrics_df = pd.DataFrame(metrics["metrics"]).T
    display_df = pd.DataFrame({
        "Model": metrics_df.index,
        "RMSE (₹)": metrics_df["RMSE_display"] if "RMSE_display" in metrics_df.columns else metrics_df["RMSE"],
        "MAE (₹)": metrics_df["MAE_display"] if "MAE_display" in metrics_df.columns else metrics_df["MAE"],
        "R² Score": metrics_df["R2"].apply(lambda v: f"{v:.4f}"),
    })

    st.table(display_df.set_index("Model"))

    best_model_key = metrics["best_model"]
    best_mae = metrics["metrics"][best_model_key].get("MAE_display", f"₹{metrics['metrics'][best_model_key]['MAE']:,.0f}")
    best_r2_val = metrics["metrics"][best_model_key]["R2"]

    st.markdown(
        f"""
        > **Optimal Model Selected:** `{best_model_key}` with an $R^2$ score of **{best_r2_val}** 
        and a Test Mean Absolute Error of **{best_mae}**.
        """
    )

    st.markdown("---")
    st.markdown("#### 🧩 5-Tier Architecture Breakdown & Interactive Step Cards")

    tier_cols = st.columns(5)
    tiers_info = [
        (
            "TIER 01",
            "💾 Data Engineering",
            "Data Engine",
            "Parametric synthesis of structural features and non-linear localized societal indicators in INR.",
            "• 5,000 Synthetic records\n• Realistic multi-tier correlation\n• Non-linear valuation formula",
        ),
        (
            "TIER 02",
            "🤖 Multi-Agent Logic",
            "Domain Agents",
            "Specialized autonomous agents assessing Infrastructure, Safety, Environment, and Growth.",
            "• 4 Domain Agents\n• Persona priority weightings\n• Dynamic pricing multipliers",
        ),
        (
            "TIER 03",
            "⚡ Predictive Models",
            "CatBoost Regressor",
            "Gradient boosted trees predicting fair baseline market property valuations.",
            "• R² = 0.9809\n• Test MAE: ₹3.53 Lakhs\n• Cross-validated hyperparameters",
        ),
        (
            "TIER 04",
            "🔬 Explainable AI",
            "SHAP TreeExplainer",
            "Decomposes black-box gradient trees into additive Rupee-exact forces per feature.",
            "• TreeExplainer integration\n• Waterfall feature attributions\n• Positive & negative drivers",
        ),
        (
            "TIER 05",
            "🎛️ Interactive SaaS",
            "Streamlit Dashboard",
            "Enterprise analytics UI with live persona toggles, radar charts, and audit exports.",
            "• Persona pill toggles\n• Instant valuation triggers\n• Valuation audit downloads",
        ),
    ]

    for col, (badge, title, subhead, desc, bullets) in zip(tier_cols, tiers_info):
        with col:
            st.markdown(
                f"""
                <div class="tier-box">
                    <div>
                        <span class="tier-badge">{badge}</span>
                        <div class="tier-heading">{title}</div>
                        <div class="tier-subhead">{subhead}</div>
                        <div class="tier-text">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Technical Specs", expanded=False):
                st.markdown(f"```text\n{bullets}\n```")
