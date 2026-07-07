# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Chronic Kidney Disease Risk Prediction System – Streamlit Application
# Run:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Ensure project root is on the path ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.preprocessing import preprocess_single_input

import utils.prediction as prediction

# We'll call functions from `prediction` at runtime to avoid import-time issues



# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(BASE_DIR, "model", "kidney_model.pkl")
TRAIN_PATH  = os.path.join(BASE_DIR, "dataset", "Training_CKD_dataset.csv")
TEST_PATH   = os.path.join(BASE_DIR, "dataset", "Testing_CKD_dataset.csv")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")

# ─── Page config ─────────────────────────────────────────────────────────────
"""st.set_page_config(
    page_title="CKD Risk Prediction System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)"""

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ────────────────────────────────────────────────────────── */
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%); }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141824 0%, #1e2535 100%);
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] .stRadio > label { color: #a0aec0; font-size: 0.85rem; }

/* ── Metric cards ──────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e2535 0%, #252d42 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,179,237,.2);
}
.metric-card .metric-value  { font-size: 2.2rem; font-weight: 700; color: #63b3ed; }
.metric-card .metric-label  { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }

/* ── Prediction result card ────────────────────────────────────────────────── */
.prediction-card {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%);
    border: 2px solid #3182ce;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 0 40px rgba(49,130,206,.25);
}
.prediction-card .stage-label { font-size: 0.9rem; color: #90cdf4; text-transform: uppercase; letter-spacing: .1em; }
.prediction-card .stage-value { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin: 8px 0; }
.prediction-card .confidence  { font-size: 1rem; color: #68d391; font-weight: 500; }

/* ── Section headers ───────────────────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #2b6cb0 0%, #3182ce 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 16px;
}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 1rem;
    font-weight: 600;
    transition: opacity .2s, transform .2s;
}
.stButton > button:hover { opacity: .9; transform: translateY(-2px); }

/* ── Streamlit default overrides ─────────────────────────────────────────── */
h1,h2,h3,h4 { color: #e2e8f0; }
p, li       { color: #a0aec0; }
.stMarkdown p { color: #cbd5e0; }
[data-testid="stMetricValue"]  { color: #63b3ed !important; }
[data-testid="stMetricLabel"]  { color: #a0aec0 !important; }

/* ── Expander ───────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader { background: #1e2535 !important; color: #e2e8f0 !important; border-radius: 8px; }

/* ── Info / Warning boxes ────────────────────────────────────────────────── */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def get_bundle():
    """Load and cache the trained model bundle using the prediction module at runtime."""
    # Debug: log prediction module info to diagnose missing-attribute issues
    try:
        print("get_bundle: prediction file=", getattr(prediction, '__file__', None))
        print("get_bundle: prediction attrs=", [a for a in dir(prediction) if 'load' in a or 'predict' in a])
    except Exception as _:
        print("get_bundle: unable to introspect prediction module")
    return prediction.load_model_bundle(MODEL_PATH)


@st.cache_data(show_spinner="Loading datasets …")
def get_datasets():
    """Load and cache training and testing CSVs."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


# Stage colours for charts
STAGE_COLORS = {
    "Healthy Kidney":        "#68d391",
    "Mild CKD (Stage 1–2)": "#f6e05e",
    "Moderate CKD (Stage 3)":"#ed8936",
    "Severe CKD (Stage 4)":  "#fc8181",
    "Kidney Failure (Stage 5)":"#e53e3e",
}


def stage_emoji(label: str) -> str:
    mapping = {
        "Healthy Kidney":        "✅",
        "Mild CKD (Stage 1–2)": "🟡",
        "Moderate CKD (Stage 3)":"🟠",
        "Severe CKD (Stage 4)":  "🔴",
        "Kidney Failure (Stage 5)":"🚨",
    }
    return mapping.get(label, "❓")


def stage_recommendation(label: str) -> str:
    recs = {
        "Healthy Kidney":
            "Great news! Your kidneys appear healthy. Maintain a balanced diet, stay hydrated, "
            "exercise regularly, and get annual check-ups.",
        "Mild CKD (Stage 1–2)":
            "Early-stage CKD detected. Your kidneys are still functioning well. Consult a nephrologist, "
            "control blood pressure & blood sugar, and limit sodium intake.",
        "Moderate CKD (Stage 3)":
            "Moderate CKD Stage 3 detected. Kidney function is moderately reduced. Immediate medical "
            "attention is advised. Follow a kidney-friendly diet and monitor labs regularly.",
        "Severe CKD (Stage 4)":
            "Severe CKD Stage 4 detected. Kidney function is severely reduced. Begin planning for renal "
            "replacement therapy (dialysis or transplant). Work closely with your care team.",
        "Kidney Failure (Stage 5)":
            "⚠️ Kidney Failure (Stage 5) detected. Immediate medical intervention is required. "
            "Dialysis or kidney transplant may be necessary. Contact your nephrologist immediately.",
    }
    return recs.get(label, "Please consult a medical professional.")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

# ── Intercept navigation requests BEFORE widgets render ──────────────────────
# Buttons on the home page write to _nav_target; we apply it to page_nav here,
# before the radio widget is constructed, so Streamlit doesn't complain.
if "_nav_target" in st.session_state and st.session_state["_nav_target"]:
    st.session_state["page_nav"] = st.session_state.pop("_nav_target")

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem;'>🫁</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:6px;'>CKD Risk System</div>
        <div style='font-size:0.75rem; color:#a0aec0; margin-top:2px;'>AI-Powered Prediction</div>
    </div>
    <hr style='border-color:#2d3748; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔬  Prediction", "ℹ️  About"],
        label_visibility="collapsed",
        key="page_nav",
    )

    st.markdown("""
    <hr style='border-color:#2d3748; margin:20px 0 12px;'>
    <div style='font-size:0.75rem; color:#718096; text-align:center;'>
        Powered by Machine Learning<br>Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Resources (with graceful error handling)
# ─────────────────────────────────────────────────────────────────────────────
model_loaded = False
bundle       = None
train_df     = None
test_df      = None

try:
    bundle       = get_bundle()
    model_loaded = True
except FileNotFoundError as e:
    pass  # handled per-page

try:
    train_df, test_df = get_datasets()
except Exception as e:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # ─── Light medical theme: override global dark styles ─────────────────
    st.markdown("""
<style>
/* Remove sidebar margin/padding to remove the empty left gutter */
[data-testid="stAppViewContainer"] {
    padding-left: 0rem !important;
}
.stApp                           { background: #f8faff !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.main .block-container {
    padding-top: 0 !important; padding-bottom: 0 !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
h1,h2,h3,h4  { color: #1a202c !important; }
.stMarkdown p { color: #718096 !important; }

/* Style the first columns layout as a top navigation bar */
div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #ffffff !important;
    box-shadow: 0 1px 12px rgba(49,130,206,.10) !important;
    padding: 16px 52px !important;
    margin-bottom: 32px !important;
    align-items: center !important;
}

/* Custom navbar button styling to make them look like text links */
div[data-testid="stHorizontalBlock"]:first-of-type button {
    background: transparent !important;
    border: none !important;
    color: #4a5568 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    padding: 6px 12px !important;
    border-radius: 0px !important;
    transition: color 0.2s !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button:hover {
    color: #3182ce !important;
    background: transparent !important;
}

/* Style the active button (Home in column 2) */
div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(2) button {
    color: #3182ce !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #3182ce !important;
}

/* Hero CTA – solid gradient pill */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#3182ce,#2b6cb0) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 13px 36px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 20px rgba(49,130,206,.35) !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(49,130,206,.45) !important;
}
</style>
""", unsafe_allow_html=True)

    # ─── NAVBAR ────────────────────────────────────────────────────────────
    _n1, _n2, _n3, _n4, _n5 = st.columns([4.0, 1.0, 1.2, 1.0, 1.0])
    with _n1:
        st.markdown("<div style='font-size:1.35rem;font-weight:700;color:#2b6cb0;padding-top:4px;white-space:nowrap;'>🫁 &nbsp;CKD Predictor</div>", unsafe_allow_html=True)
    with _n2:
        st.button("Home", key="nh")
    with _n3:
        if st.button("Prediction", key="np"):
            st.session_state["_nav_target"] = "🔬  Prediction"; st.rerun()
    with _n4:
        if st.button("About", key="na"):
            st.session_state["_nav_target"] = "ℹ️  About"; st.rerun()
    with _n5:
        st.button("Contact", key="nc")

    # ─── HERO ─────────────────────────────────────────────────────────────
    _hl, _hr = st.columns([1.1, 0.9])

    with _hl:
        st.markdown("""
<div style='padding:55px 20px 20px 50px;'>
  <div style='display:inline-block;background:#ebf4ff;color:#2b6cb0;
              font-size:.77rem;font-weight:600;padding:5px 14px;
              border-radius:20px;margin-bottom:18px;
              text-transform:uppercase;letter-spacing:.07em;'>
    &#127973; AI-Powered Healthcare
  </div>
  <h1 style='font-size:2.7rem;font-weight:800;color:#1a202c !important;
             line-height:1.2;margin-bottom:18px;'>
    Chronic Kidney Disease<br>
    <span style='color:#3182ce;'>Prediction</span>
  </h1>
  <p style='font-size:1.05rem;color:#718096 !important;line-height:1.85;
            max-width:490px;margin-bottom:0;'>
    An AI-powered system that helps predict the risk of Chronic Kidney Disease
    using clinical parameters.
  </p>
</div>
""", unsafe_allow_html=True)
        _pb1, _pb2, _pb3 = st.columns([0.5, 2.2, 1.0])
        with _pb2:
            if st.button("🔬  Start Prediction", key="hero_cta",
                         type="primary", use_container_width=True):
                st.session_state["_nav_target"] = "🔬  Prediction"
                st.rerun()

    with _hr:
        st.markdown("""
<div style='display:flex;justify-content:center;align-items:center;
            padding:45px 15px 30px;min-height:310px;'>
<svg viewBox="0 0 420 380" xmlns="http://www.w3.org/2000/svg" width="360" height="320">
  <circle cx="210" cy="185" r="168" fill="#dbeafe" opacity=".28"/>
  <circle cx="210" cy="185" r="130" fill="#bfdbfe" opacity=".20"/>
  <circle cx="58"  cy="68"  r="9"  fill="#93c5fd" opacity=".50"/>
  <circle cx="364" cy="78"  r="13" fill="#60a5fa" opacity=".32"/>
  <circle cx="46"  cy="296" r="10" fill="#93c5fd" opacity=".38"/>
  <circle cx="372" cy="304" r="7"  fill="#60a5fa" opacity=".38"/>
  <circle cx="210" cy="26"  r="7"  fill="#3b82f6" opacity=".26"/>
  <g transform="translate(72,78)">
    <path d="M52 18 C20 18 4 54 4 92 C4 130 19 160 46 170
             C62 176 78 165 83 149 C92 123 90 96 87 70
             C84 44 72 18 52 18 Z" fill="#3b82f6" opacity=".82"/>
    <path d="M34 42 C22 58 18 80 22 100 C26 118 36 134 48 142
             C36 126 30 104 32 84 C34 66 34 50 34 42 Z"
          fill="#93c5fd" opacity=".34"/>
    <ellipse cx="62" cy="97" rx="17" ry="23" fill="#bfdbfe" opacity=".50"/>
    <path d="M64 168 Q68 190 70 210" stroke="#2563eb" stroke-width="5"
          fill="none" stroke-linecap="round"/>
  </g>
  <g transform="translate(252,78)">
    <path d="M44 18 C76 18 92 54 92 92 C92 130 77 160 50 170
             C34 176 18 165 13 149 C4 123 6 96 9 70
             C12 44 24 18 44 18 Z" fill="#60a5fa" opacity=".88"/>
    <path d="M62 42 C74 58 78 80 74 100 C70 118 60 134 48 142
             C60 126 66 104 64 84 C62 66 62 50 62 42 Z"
          fill="#bfdbfe" opacity=".34"/>
    <ellipse cx="34" cy="97" rx="17" ry="23" fill="#dbeafe" opacity=".50"/>
    <path d="M32 168 Q28 190 26 210" stroke="#2563eb" stroke-width="5"
          fill="none" stroke-linecap="round"/>
  </g>
  <ellipse cx="210" cy="298" rx="42" ry="30" fill="#60a5fa" opacity=".50"/>
  <path d="M148 288 Q170 296 210 298" stroke="#2563eb" stroke-width="3.5"
        fill="none" stroke-linecap="round"/>
  <path d="M272 288 Q250 296 210 298" stroke="#2563eb" stroke-width="3.5"
        fill="none" stroke-linecap="round"/>
  <rect x="330" y="50" width="44" height="44" rx="10" fill="#eff6ff"/>
  <rect x="340" y="57" width="24" height="10" rx="3" fill="#3b82f6" opacity=".78"/>
  <rect x="347" y="50" width="10" height="37" rx="3" fill="#3b82f6" opacity=".78"/>
  <path d="M28 355 L60 355 L74 332 L90 378 L106 346 L120 355 L390 355"
        stroke="#3b82f6" stroke-width="2.5" fill="none" opacity=".36"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>
""", unsafe_allow_html=True)


    # ─── WHY EARLY DETECTION ──────────────────────────────────────────────

    _il, _ir = st.columns([1.1, 0.9])

    with _il:
        st.markdown("""
<div style='padding:55px 20px 40px 50px;'>
  <div style='display:inline-block;background:#ebf4ff;color:#2b6cb0;
              font-size:.75rem;font-weight:600;padding:5px 14px;
              border-radius:20px;margin-bottom:14px;
              text-transform:uppercase;letter-spacing:.07em;'>
    &#128161; Early Detection
  </div>
  <h2 style='font-size:1.85rem;font-weight:700;color:#1a202c !important;
             margin-bottom:14px;line-height:1.3;'>
    Why Early Detection<br>Matters
  </h2>
  <p style='font-size:.97rem;color:#718096 !important;line-height:1.85;margin-bottom:14px;'>
    Early diagnosis of Chronic Kidney Disease helps patients receive timely treatment,
    slow disease progression, and improve long-term health outcomes.
  </p>
  <p style='font-size:.97rem;color:#718096 !important;line-height:1.85;'>
    Our AI-powered system analyses clinical biomarkers to identify CKD risk at its
    earliest stages &#8212; when intervention is most effective and outcomes are
    significantly better.
  </p>
</div>
""", unsafe_allow_html=True)

    with _ir:
        st.markdown("""
<div style='display:flex;justify-content:center;align-items:center;padding:40px 15px;'>
<svg viewBox="0 0 360 280" xmlns="http://www.w3.org/2000/svg" width="320" height="248">
  <rect x="8" y="8" width="344" height="264" rx="22" fill="#eff6ff" opacity=".62"/>
  <circle cx="104" cy="78" r="31" fill="#bfdbfe"/>
  <path d="M62 132 Q60 112 104 112 Q148 112 146 132 L140 215 L68 215 Z"
        fill="#3b82f6" opacity=".70"/>
  <path d="M86 118 L98 158 L104 118 Z" fill="white" opacity=".52"/>
  <path d="M122 118 L110 158 L104 118 Z" fill="white" opacity=".52"/>
  <path d="M120 145 Q143 153 147 173 Q149 184 142 188"
        stroke="#1e40af" stroke-width="3" fill="none" stroke-linecap="round"/>
  <circle cx="140" cy="190" r="9" fill="#1e40af" opacity=".52"/>
  <rect x="174" y="75" width="158" height="182" rx="14"
        fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
  <rect x="174" y="75" width="158" height="30" rx="14" fill="#3b82f6" opacity=".10"/>
  <text x="253" y="95" text-anchor="middle" font-size="11" fill="#1d4ed8"
        font-family="Inter,sans-serif" font-weight="600">Patient Report</text>
  <rect x="190" y="123" width="126" height="7" rx="3.5" fill="#bfdbfe" opacity=".78"/>
  <rect x="190" y="140" width="100" height="7" rx="3.5" fill="#93c5fd" opacity=".62"/>
  <rect x="190" y="157" width="114" height="7" rx="3.5" fill="#bfdbfe" opacity=".72"/>
  <polyline points="190,222 208,208 224,216 242,196 258,206 275,184 292,192 315,174"
            stroke="#3b82f6" stroke-width="2.5" fill="none"
            stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M190,222 208,208 224,216 242,196 258,206 275,184 292,192 315,174 315,248 190,248 Z"
        fill="#3b82f6" opacity=".06"/>
  <circle cx="306" cy="123" r="11" fill="#dcfce7"/>
  <polyline points="300,123 305,129 312,117" stroke="#16a34a"
            stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <circle cx="306" cy="150" r="11" fill="#dcfce7"/>
  <polyline points="300,150 305,156 312,144" stroke="#16a34a"
            stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="M14 255 L38 255 L48 238 L62 272 L74 248 L84 255 L110 255"
        stroke="#60a5fa" stroke-width="2" fill="none" opacity=".48"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>
""", unsafe_allow_html=True)

    # ─── DISCLAIMER ───────────────────────────────────────────────────────
    st.markdown("""
<div style='margin:8px 44px 28px;padding:18px 26px;background:#fffbeb;
            border:1px solid #fbd38d;border-radius:14px;
            display:flex;align-items:flex-start;gap:14px;'>
  <span style='font-size:1.3rem;flex-shrink:0;margin-top:2px;'>&#9888;&#65039;</span>
  <div style='font-size:.87rem;color:#92400e !important;line-height:1.65;'>
    <strong>Medical Disclaimer:</strong> This application is intended for
    <strong>educational purposes only</strong> and should not replace professional
    medical advice. Always consult a qualified healthcare provider for medical decisions.
  </div>
</div>
""", unsafe_allow_html=True)



# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Prediction":
    st.markdown("<h1 style='color:#e2e8f0;'>🔬 CKD Stage Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0aec0;'>Fill in the patient's clinical details below and press <b>Predict</b>.</p>",
                unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ Model not loaded. Run `python train_model.py` first.")
        st.stop()

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        # ── Demographics & Vitals ──────────────────────────────────────────
        st.markdown("<div class='section-header'>👤 Demographics & Vital Signs</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age       = c1.number_input("Age (years)", 1, 120, 50, help="Patient age in years")
        gender    = c2.selectbox("Gender", ["Male (1)", "Female (0)"])
        bmi       = c3.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
        heart_rate= c4.number_input("Heart Rate (bpm)", 30, 200, 75)

        c1, c2, c3, c4 = st.columns(4)
        sys_bp    = c1.number_input("Systolic BP (mmHg)",  60, 220, 120)
        dia_bp    = c2.number_input("Diastolic BP (mmHg)", 40, 140, 80)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Kidney Function Markers ────────────────────────────────────────
        st.markdown("<div class='section-header'>🧪 Kidney Function Markers</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        creatinine  = c1.number_input("Serum Creatinine (mg/dL)", 0.0, 20.0, 1.0, step=0.1)
        bun         = c2.number_input("Blood Urea Nitrogen (mg/dL)", 0, 200, 15)
        egfr        = c3.number_input("eGFR (mL/min/1.73m²)", 0, 200, 90)
        urine_alb   = c4.number_input("Urine Albumin (mg/L)", 0, 5000, 20)

        c1, c2, c3, c4 = st.columns(4)
        urine_prot  = c1.number_input("Urine Protein (mg/dL)", 0, 1000, 5)
        acr         = c2.number_input("Albumin-Creatinine Ratio", 0, 3000, 30)
        usg         = c3.number_input("Urine Specific Gravity", 1.000, 1.040, 1.015, step=0.001, format="%.3f")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Electrolytes ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>⚗️ Electrolytes & Minerals</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        sodium   = c1.number_input("Sodium (mEq/L)",     100, 180, 140)
        potassium= c2.number_input("Potassium (mEq/L)",  2.0, 8.0, 4.0, step=0.1)
        calcium  = c3.number_input("Calcium (mg/dL)",    4.0, 14.0, 9.5, step=0.1)
        phosphorus=c4.number_input("Phosphorus (mg/dL)", 1.0, 12.0, 3.5, step=0.1)

        c1, c2 = st.columns(2)
        chloride    = c1.number_input("Chloride (mEq/L)",   80, 130, 102)
        bicarbonate = c2.number_input("Bicarbonate (mEq/L)",10, 40,  24)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Blood Panel ───────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🩸 Blood Panel</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        hemoglobin = c1.number_input("Hemoglobin (g/dL)",     4.0, 20.0, 14.0, step=0.1)
        rbc        = c2.number_input("RBC Count (×10⁶/µL)",  2.0, 8.0,  5.0,  step=0.1)
        wbc        = c3.number_input("WBC Count (/µL)",       1000, 30000, 7000, step=100)
        platelets  = c4.number_input("Platelet Count (/µL)",  50000, 800000, 250000, step=1000)

        c1, c2 = st.columns(2)
        pcv = c1.number_input("Packed Cell Volume (%)", 10, 60, 42)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Glucose & Lipids ──────────────────────────────────────────────
        st.markdown("<div class='section-header'>🍬 Glucose & Lipids</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        bgr       = c1.number_input("Blood Glucose Random (mg/dL)", 50, 500, 100)
        fasting_g = c2.number_input("Fasting Glucose (mg/dL)",      60, 400, 90)
        hba1c     = c3.number_input("HbA1c (%)",                    3.0, 15.0, 5.5, step=0.1)
        cholesterol=c4.number_input("Cholesterol (mg/dL)",           80, 400, 180)

        c1, c2 = st.columns(2)
        triglycerides = c1.number_input("Triglycerides (mg/dL)", 30, 1000, 150)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Proteins ──────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔬 Serum Proteins</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        serum_albumin = c1.number_input("Serum Albumin (g/dL)",  1.0, 6.0, 4.0, step=0.1)
        total_protein = c2.number_input("Total Protein (g/dL)",  3.0, 10.0, 7.0, step=0.1)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Factors ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>⚠️ Risk Factors</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        diabetes    = c1.selectbox("Diabetes",              ["Yes", "No"])
        hypertension= c2.selectbox("Hypertension",          ["Yes", "No"])
        smoking     = c3.selectbox("Smoking Status",         ["Yes", "No"])
        family_hist = c4.selectbox("Family History Kidney",  ["Yes", "No"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict CKD Stage", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────
    if submitted:
        gender_val = 1 if "Male" in gender else 0

        input_dict = {
            "Age": age, "Gender": gender_val, "BMI": bmi,
            "Systolic_BP": sys_bp, "Diastolic_BP": dia_bp, "Heart_Rate": heart_rate,
            "Serum_Creatinine": creatinine, "Blood_Urea_Nitrogen": bun, "eGFR": egfr,
            "Urine_Albumin": urine_alb, "Urine_Protein": urine_prot,
            "Albumin_Creatinine_Ratio": acr, "Urine_Specific_Gravity": usg,
            "Sodium": sodium, "Potassium": potassium, "Calcium": calcium,
            "Phosphorus": phosphorus, "Chloride": chloride, "Bicarbonate": bicarbonate,
            "Hemoglobin": hemoglobin, "RBC_Count": rbc, "WBC_Count": wbc,
            "Platelet_Count": platelets, "Packed_Cell_Volume": pcv,
            "Blood_Glucose_Random": bgr, "Fasting_Glucose": fasting_g, "HbA1c": hba1c,
            "Cholesterol": cholesterol, "Triglycerides": triglycerides,
            "Serum_Albumin": serum_albumin, "Total_Protein": total_protein,
            "Diabetes": diabetes, "Hypertension": hypertension,
            "Smoking_Status": smoking, "Family_History_Kidney": family_hist,
        }

        try:
            input_df = preprocess_single_input(input_dict, bundle["encoders"], bundle["scaler"])
            label, proba = prediction.predict_single(input_df, bundle)
            emoji = stage_emoji(label)
            rec   = stage_recommendation(label)

            conf_text = ""
            if proba is not None:
                conf = proba.max() * 100
                conf_text = f"<div class='confidence'>Confidence: {conf:.1f}%</div>"

            st.markdown(f"""
            <div class='prediction-card'>
                <div class='stage-label'>Predicted CKD Stage</div>
                <div class='stage-value'>{emoji} {label}</div>
                {conf_text}
            </div>""", unsafe_allow_html=True)

            # Recommendation
            st.info(f"**Clinical Recommendation:** {rec}")

            # Probability breakdown
            if proba is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📊 Prediction Confidence per Stage")
                classes = bundle["classes"]
                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_facecolor("#1e2535")
                ax.set_facecolor("#1e2535")
                colors = [STAGE_COLORS.get(c, "#63b3ed") for c in classes]
                bars = ax.barh(classes, proba * 100, color=colors, height=0.5, edgecolor="none")
                ax.set_xlabel("Probability (%)", color="#a0aec0")
                ax.set_xlim(0, 100)
                ax.tick_params(colors="#a0aec0")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                for bar, p in zip(bars, proba):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                            f"{p*100:.1f}%", va="center", color="#e2e8f0", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        except Exception as ex:
            st.error(f"❌ Prediction error: {ex}")





# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("<h1 style='color:#e2e8f0;'>ℹ️ About This System</h1>", unsafe_allow_html=True)

    # CKD Overview
    st.markdown("<div class='section-header'>🫁 What is Chronic Kidney Disease?</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:#1e2535; border-radius:12px; padding:24px; border:1px solid #2d3748; color:#cbd5e0; line-height:1.8;'>

**Chronic Kidney Disease (CKD)** is a long-term condition where the kidneys gradually lose their
filtering ability over months or years. The kidneys play a critical role in removing waste products
and excess fluid from the blood, regulating blood pressure, and maintaining electrolyte balance.

CKD is classified into five stages based on **eGFR (estimated Glomerular Filtration Rate)**:

| Stage | eGFR Range        | Description                  |
|-------|-------------------|------------------------------|
| 1–2   | ≥ 60 mL/min       | Mild – kidney damage present |
| 3     | 30–59 mL/min      | Moderate reduction           |
| 4     | 15–29 mL/min      | Severe reduction             |
| 5     | < 15 mL/min       | Kidney failure               |

**Risk factors** include diabetes, hypertension, family history, obesity, and smoking.
Early detection and management can significantly slow disease progression.
</div>
""", unsafe_allow_html=True)


    # Disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer**: This application is intended for **educational and research purposes only**.
    It is **not a substitute** for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical decisions.
    """)
