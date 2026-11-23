"""Streamlit app for market-value prediction."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

# ---------- Password gate ----------
try:
    app_password = st.secrets.get("APP_PASSWORD")
except Exception:
    app_password = os.getenv("APP_PASSWORD")
if app_password:
    if not st.session_state.get("auth"):
        st.markdown(
            "<div style='text-align:center;margin-top:80px'>"
            "<h1>🔒 Football Market Value Predictor</h1>"
            "<p style='color:#666'>Enter password to continue</p></div>",
            unsafe_allow_html=True,
        )
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == app_password:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

# ---------- Model download ----------
MODEL_PATH = os.path.join("models", "lgbm_market_value.joblib")
try:
    model_url = st.secrets.get("MODEL_URL")
except Exception:
    model_url = os.getenv("MODEL_URL")
if not os.path.exists(MODEL_PATH) and model_url:
    os.makedirs("models", exist_ok=True)
    with st.spinner("Downloading model (50MB)..."):
        urllib.request.urlretrieve(model_url, MODEL_PATH)
    st.success("Model downloaded.")
elif not os.path.exists(MODEL_PATH):
    st.error("Model not found. Commit models/lgbm_market_value.joblib or set MODEL_URL secret.")
    st.stop()

os.environ.setdefault("MODEL_PATH", MODEL_PATH)

from predict import predict_df

# ---------- Page config ----------
st.set_page_config(
    page_title="Football Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #555; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .card {
        background: #f8f9fb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        color: white;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 4px 14px rgba(31,119,180,0.3);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚽ About")
    st.markdown(
        "Predicts a football player's transfer market value (EUR) using a "
        "tuned **LightGBM** model trained on Transfermarkt data."
    )
    st.markdown("---")
    st.markdown("**Model metrics (test, 2024+):**")
    st.markdown("- R² = 0.975")
    st.markdown("- MAE = €420k")
    st.markdown("- MedAE = €84k")
    st.markdown("- Within 20% = 67.6%")
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown(
        "- Fill in history fields (prev/peak value) for best accuracy.\n"
        "- Cold-start (no history) is weaker (~R² 0.61).\n"
        "- Values are estimates, not actual transfer fees."
    )
    st.markdown("---")
    st.caption("Data: Transfermarkt via HuggingFace mirror")

# ---------- Header ----------
st.markdown('<div class="main-title">⚽ Football Market Value Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estimate a player\'s transfer market value from profile, '
    'league, contract and performance history.</div>',
    unsafe_allow_html=True,
)

# ---------- Defaults ----------
defaults = {
    "age": 24.0, "height": 180.0, "caps": 0.0, "goals": 0.0,
    "position": "Attack", "sub_position": "Centre-Forward", "foot": "right",
    "country": "Germany", "league": "GB1", "contract_months": 12.0,
    "cum_matches": 0.0, "cum_goals": 0.0, "cum_assists": 0.0, "cum_minutes": 0.0,
    "num_transfers": 0.0, "prev_value": 0.0, "peak_so_far": 0.0,
    "days_since_prev": 9999.0, "player_highest_mv": 0.0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------- Example button ----------
if st.button("✨ Load example (Bukayo Saka)", use_container_width=True):
    st.session_state.update({
        "age": 24.5, "height": 178.0, "caps": 54.0, "goals": 14.0,
        "position": "Attack", "sub_position": "Right Winger", "foot": "left",
        "country": "England", "league": "GB1", "contract_months": 51.7,
        "cum_matches": 290.0, "cum_goals": 78.0, "cum_assists": 72.0,
        "cum_minutes": 21857.0, "num_transfers": 4.0, "prev_value": 130000000.0,
        "peak_so_far": 150000000.0, "days_since_prev": 90.0,
        "player_highest_mv": 150000000.0,
    })
    st.rerun()

# ---------- Inputs ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Profile")
    age = st.number_input("Age", min_value=15.0, max_value=45.0, key="age")
    height = st.number_input("Height (cm)", min_value=150.0, max_value=210.0, key="height")
    caps = st.number_input("International caps", min_value=0.0, key="caps")
    goals = st.number_input("International goals", min_value=0.0, key="goals")
    position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"], key="position")
    sub_position = st.text_input("Sub-position", key="sub_position")
    foot = st.selectbox("Foot", ["right", "left", "both"], key="foot")

with col2:
    st.markdown("#### Context & history")
    country = st.text_input("Country of citizenship", key="country")
    league = st.text_input("League id", key="league")
    contract_months = st.number_input("Contract months remaining", key="contract_months")
    cum_matches = st.number_input("Cumulative matches", key="cum_matches")
    cum_goals = st.number_input("Cumulative goals", key="cum_goals")
    cum_assists = st.number_input("Cumulative assists", key="cum_assists")
    cum_minutes = st.number_input("Cumulative minutes", key="cum_minutes")
    num_transfers = st.number_input("Number of transfers", key="num_transfers")
    prev_value = st.number_input("Previous market value (EUR)", key="prev_value")
    peak_so_far = st.number_input("Peak market value so far (EUR)", key="peak_so_far")
    days_since_prev = st.number_input("Days since previous valuation", key="days_since_prev")
    player_highest_mv = st.number_input("Player highest market value (EUR)", key="player_highest_mv")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Predict ----------
if st.button("🔮 Predict market value", use_container_width=True, type="primary"):
    row = pd.DataFrame([{
        "age": age, "height_in_cm": height, "international_caps": caps,
        "international_goals": goals, "position": position,
        "sub_position": sub_position, "foot": foot,
        "country_of_citizenship": country, "league": league,
        "contract_months": contract_months, "cum_matches": cum_matches,
        "cum_goals": cum_goals, "cum_assists": cum_assists,
        "cum_minutes": cum_minutes, "num_transfers": num_transfers,
        "prev_value": prev_value, "peak_so_far": peak_so_far,
        "days_since_prev": days_since_prev, "player_highest_mv": player_highest_mv,
    }])
    val = predict_df(row)[0]
    st.markdown(f'<div class="result-box">EUR {val:,.0f}</div>', unsafe_allow_html=True)

# ---------- Info ----------
with st.expander("ℹ️ How it works & limitations"):
    st.markdown(
        "- Trained on Transfermarkt historical valuations (2003–2026).\n"
        "- Uses profile + league + contract + cumulative performance + value history.\n"
        "- Test R²=0.975 (2024+ valuations) when history is provided.\n"
        "- Cold-start (no history) is weaker (~R²=0.61).\n"
        "- Market values are estimates, not actual transfer fees.\n"
        "- Data frozen mid-2026; retrain for newer seasons."
    )

st.caption("Model: tuned LightGBM on Transfermarkt data. Built with Streamlit.")