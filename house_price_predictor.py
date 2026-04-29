"""
House Price Predictor — Polynomial Ridge Regression
Run: streamlit run house_price_predictor.py

pip install streamlit pandas numpy scikit-learn openpyxl
Place house_pricing_dataset_2000.xlsx in the same folder as this script.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES    = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_Years"]
TARGET      = "Price_USD"
POLY_DEGREE = 2
RIDGE_ALPHA = 1.0

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "house_pricing_dataset_2000.xlsx",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — THEME-AGNOSTIC (works identically in light & dark mode)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Root variables ── */
  :root {
    --gold:    #C9A84C;
    --gold2:   #E8C97A;
    --cream:   #FDF6E3;
    --dark:    #0B1120;
    --panel:   rgba(11,17,32,0.82);
    --border:  rgba(201,168,76,0.35);
    --muted:   rgba(253,246,227,0.55);
  }

  /* ── Force dark background on Streamlit root — overrides both themes ── */
  .stApp,
  .stApp > div,
  section[data-testid="stAppViewContainer"],
  div[data-testid="stAppViewBlockContainer"] {
    background-color: #0B1120 !important;
    background:
      linear-gradient(
        to bottom,
        rgba(11,17,32,0.72) 0%,
        rgba(11,17,32,0.58) 40%,
        rgba(11,17,32,0.80) 100%
      ),
      url("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1600&auto=format&fit=crop&q=80")
      center center / cover no-repeat fixed !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--cream) !important;
  }

  /* ── Force ALL text inside app to cream — kills Streamlit theme inheritance ── */
  .stApp p,
  .stApp span,
  .stApp label,
  .stApp div,
  .stApp h1, .stApp h2, .stApp h3,
  .stApp li, .stApp td, .stApp th {
    color: var(--cream) !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    max-width: 720px !important;
    padding: 2.5rem 2rem 4rem !important;
  }

  /* ── Hero container ── */
  .hero-wrap {
    text-align: center;
    padding: 3rem 0 2rem;
    animation: fadeDown 0.8s ease both;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 3.8rem);
    font-weight: 900;
    color: var(--cream) !important;
    line-height: 1.1;
    margin: 0;
    text-shadow: 0 5px 30px rgba(0,0,0,0.6);
  }
  .gradient-text {
    background: linear-gradient(90deg, #f4d87a, #c9a84c, #f4d87a);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
  }
  .hero-title::after {
    content: "";
    display: block;
    width: 80px;
    height: 3px;
    margin: 14px auto 0;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    animation: expand 1.2s ease forwards;
  }

  /* ── Stats ribbon ── */
  .stats-ribbon {
    display: flex;
    gap: 0;
    background: rgba(201,168,76,0.07);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    margin: 1.6rem 0 2rem;
    animation: fadeUp 0.8s 0.2s ease both;
  }
  .stat-item {
    flex: 1;
    padding: 14px 8px;
    text-align: center;
    border-right: 1px solid var(--border);
  }
  .stat-item:last-child { border-right: none; }
  .stat-val {
    font-size: 20px;
    font-weight: 700;
    color: var(--gold2) !important;
    font-family: 'Playfair Display', serif;
  }
  .stat-lbl {
    font-size: 9.5px;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
  }

  /* ── Section title ── */
  .card-section-title {
    font-size: 11px;
    color: var(--gold) !important;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 600;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ══════════════════════════════════════════════════════
     INPUT OVERRIDES — forced for BOTH light & dark mode
     Use !important on every property so Streamlit's
     theme stylesheet (which loads after ours) cannot win.
  ══════════════════════════════════════════════════════ */

  /* Label text */
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSlider"] label {
    color: var(--cream) !important;
    -webkit-text-fill-color: var(--cream) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
  }

  /* Input box */
  div[data-testid="stNumberInput"] input {
    background: rgba(20, 30, 55, 0.85) !important;
    border: 1px solid rgba(201,168,76,0.30) !important;
    border-radius: 10px !important;
    color: var(--cream) !important;
    -webkit-text-fill-color: var(--cream) !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    caret-color: var(--gold2) !important;
    transition: border-color 0.2s;
  }
  div[data-testid="stNumberInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
  }

  /* +/- stepper buttons */
  div[data-testid="stNumberInput"] button {
    background: rgba(201,168,76,0.12) !important;
    border-color: rgba(201,168,76,0.25) !important;
    color: var(--gold2) !important;
  }
  div[data-testid="stNumberInput"] button svg {
    fill: var(--gold2) !important;
    stroke: var(--gold2) !important;
  }

  /* Entire number input container background */
  div[data-testid="stNumberInput"] > div {
    background: transparent !important;
  }

  /* ── Primary button ── */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #C9A84C 0%, #E8C97A 55%, #C9A84C 100%) !important;
    background-size: 200% 200% !important;
    color: #0B1120 !important;
    -webkit-text-fill-color: #0B1120 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    height: 56px !important;
    box-shadow: 0 6px 28px rgba(201,168,76,0.35) !important;
    transition: all 0.25s ease !important;
    animation: shimmer 3s linear infinite !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(201,168,76,0.50) !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
  }

  /* ── Warning override ── */
  div[data-testid="stAlert"] {
    background: rgba(201,168,76,0.10) !important;
    border: 1px solid rgba(201,168,76,0.40) !important;
    border-radius: 10px !important;
    color: var(--gold2) !important;
  }
  div[data-testid="stAlert"] p {
    color: var(--gold2) !important;
    -webkit-text-fill-color: var(--gold2) !important;
  }

  /* ── Result card ── */
  .result-card {
    background: linear-gradient(135deg, rgba(11,17,32,0.95) 0%, rgba(18,30,58,0.95) 100%);
    border: 1px solid var(--gold);
    border-radius: 20px;
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-top: 1.6rem;
    box-shadow: 0 0 60px rgba(201,168,76,0.18), 0 20px 60px rgba(0,0,0,0.5);
    animation: popIn 0.55s cubic-bezier(0.34,1.56,0.64,1) both;
    position: relative;
    overflow: hidden;
  }
  .result-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(201,168,76,0.07) 0%, transparent 65%);
    pointer-events: none;
  }
  .result-label {
    font-size: 10.5px;
    color: var(--gold) !important;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 8vw, 4.2rem);
    font-weight: 900;
    color: var(--cream) !important;
    line-height: 1.1;
    text-shadow: 0 0 40px rgba(201,168,76,0.4);
  }
  .result-price span { color: var(--gold2) !important; }
  .result-range {
    font-size: 13px;
    color: var(--muted) !important;
    margin-top: 8px;
    font-weight: 300;
    letter-spacing: 0.3px;
  }

  /* ── Metric pills ── */
  .metrics-row {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    justify-content: center;
    flex-wrap: wrap;
  }
  .metric-pill {
    background: rgba(201,168,76,0.09);
    border: 1px solid rgba(201,168,76,0.30);
    border-radius: 12px;
    padding: 10px 20px;
    text-align: center;
    min-width: 90px;
  }
  .metric-pill .mp-val {
    font-size: 18px;
    font-weight: 700;
    color: var(--gold2) !important;
    font-family: 'Playfair Display', serif;
  }
  .metric-pill .mp-lbl {
    font-size: 9.5px;
    color: var(--muted) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* ── Confidence bar ── */
  .conf-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    height: 7px;
    width: 100%;
    margin-top: 18px;
  }
  .conf-bar-fill {
    background: linear-gradient(90deg, #C9A84C, #E8C97A);
    border-radius: 8px;
    height: 7px;
    box-shadow: 0 0 12px rgba(201,168,76,0.5);
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
  }
  .conf-label {
    font-size: 11px;
    color: var(--muted) !important;
    margin-top: 6px;
    font-weight: 300;
  }

  /* ── Divider ── */
  hr {
    border-color: rgba(201,168,76,0.20) !important;
    margin: 1.4rem 0 !important;
  }

  /* ── Expander ── */
  div[data-testid="stExpander"] {
    background: rgba(11,17,32,0.60) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
  }
  div[data-testid="stExpander"] summary,
  div[data-testid="stExpander"] summary span {
    color: var(--gold2) !important;
    -webkit-text-fill-color: var(--gold2) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
  }
  div[data-testid="stExpander"] p,
  div[data-testid="stExpander"] li,
  div[data-testid="stExpander"] td {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 13px !important;
  }
  div[data-testid="stExpander"] th {
    color: var(--gold2) !important;
    -webkit-text-fill-color: var(--gold2) !important;
  }

  /* ── Keyframes ── */
  @keyframes fadeDown {
    from { opacity:0; transform: translateY(-22px); }
    to   { opacity:1; transform: translateY(0); }
  }
  @keyframes fadeUp {
    from { opacity:0; transform: translateY(18px); }
    to   { opacity:1; transform: translateY(0); }
  }
  @keyframes popIn {
    from { opacity:0; transform: scale(0.90); }
    to   { opacity:1; transform: scale(1); }
  }
  @keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  @keyframes shine {
    0%   { background-position: 0%; }
    100% { background-position: 200%; }
  }
  @keyframes expand {
    from { width: 0; opacity: 0; }
    to   { width: 80px; opacity: 1; }
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on 2,000 records…")
def get_trained_model():
    df = pd.read_excel(DATASET_PATH, sheet_name="House Pricing Data", header=1)
    df.columns = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_Years", "Price_USD"]
    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("poly",   PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=RIDGE_ALPHA)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)
    cv_r2   = cross_val_score(pipeline, X, y, cv=5, scoring="r2").mean()
    n_feat  = pipeline.named_steps["poly"].n_output_features_

    return pipeline, round(test_r2 * 100, 2), round(cv_r2 * 100, 2), int(mae), len(df), n_feat


def predict_price(pipeline, area, bedrooms, bathrooms, age):
    X_new = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms":  bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
    }])
    raw = float(pipeline.predict(X_new)[0])
    raw = max(raw, 50_000)
    return round(raw / 1000) * 1000


def confidence_score(area, bedrooms, bathrooms, age):
    center = {"area": 2320, "bed": 3.07, "bath": 1.72, "age": 24.8}
    scale  = {"area": 987,  "bed": 1.06, "bath": 0.68, "age": 14.4}
    z = (
        abs(area      - center["area"]) / scale["area"]
        + abs(bedrooms  - center["bed"])  / scale["bed"]
        + abs(bathrooms - center["bath"]) / scale["bath"]
        + abs(age       - center["age"])  / scale["age"]
    ) / 4
    return max(60, min(99, int(99 - z * 18)))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
try:
    pipeline, test_acc, cv_acc, mae, n_rows, n_features = get_trained_model()
except FileNotFoundError:
    st.error(
        "❌ Dataset not found.\n\n"
        "Place **house_pricing_dataset_2000.xlsx** in the same folder as this script."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# UI — HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <h1 class="hero-title">
    <span> House Price</span><br>
    <span class="gradient-text">Predictor</span>
  </h1>
</div>
""", unsafe_allow_html=True)

# ── Stats ribbon ──
st.markdown(f"""
<div class="stats-ribbon">
  <div class="stat-item">
    <div class="stat-val">{n_rows:,}</div>
    <div class="stat-lbl">🏘️ Records</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{test_acc}%</div>
    <div class="stat-lbl">🎯 Test R²</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{cv_acc}%</div>
    <div class="stat-lbl">📊 CV R²</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">${mae:,}</div>
    <div class="stat-lbl">📉 Avg Error</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card-section-title">🏠 Property Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    area      = st.number_input(" Area (sqft)",          min_value=100,  max_value=10000, value=2000, step=50)
    bathrooms = st.number_input(" Bathrooms",             min_value=1,    max_value=8,     value=2,    step=1)
with col2:
    bedrooms  = st.number_input(" Bedrooms",             min_value=1,    max_value=10,    value=3,    step=1)
    age       = st.number_input(" Age of House (Years)", min_value=0,    max_value=100,   value=10,   step=1)

if bathrooms > bedrooms:
    st.warning("⚠️ Bathrooms exceed bedrooms — please double-check your inputs.")

st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("🔮 Predict Market Value", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# UI — RESULT
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    price  = predict_price(pipeline, area, bedrooms, bathrooms, age)
    conf   = confidence_score(area, bedrooms, bathrooms, age)

    spread = max(0.03, (1 - conf / 100) * 0.20)
    low    = int(round(price * (1 - spread) / 1000) * 1000)
    high   = int(round(price * (1 + spread) / 1000) * 1000)

    price_int = f"{price:,.0f}"

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">✦ Estimated Market Value ✦</div>
      <div class="result-price"><span>$</span>{price_int}</div>
      <div class="result-range">📊 Likely range &nbsp;·&nbsp; ${low:,.0f} – ${high:,.0f}</div>

      <div class="metrics-row">
        <div class="metric-pill">
          <div class="mp-val">{test_acc}%</div>
          <div class="mp-lbl">🎯 Model R²</div>
        </div>
        <div class="metric-pill">
          <div class="mp-val">${mae:,}</div>
          <div class="mp-lbl">📉 Avg Error</div>
        </div>
        <div class="metric-pill">
          <div class="mp-val">{conf}%</div>
          <div class="mp-lbl">✅ Confidence</div>
        </div>
      </div>

      <div class="conf-bar-wrap">
        <div class="conf-bar-fill" style="width:{conf}%;"></div>
      </div>
      <div class="conf-label">Prediction confidence based on similarity to training data</div>
    </div>
    """, unsafe_allow_html=True)