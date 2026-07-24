import os
import sys
from datetime import datetime, timedelta, date
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import altair as alt

# Path resolution to import src modules and weather functions
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR) if os.path.basename(APP_DIR) == "app" else APP_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.weather as weather_module
from src.predict import predict_delay, predict_batch, selected_features
from src.weather import get_live_weather_prediction, get_weather_forecast, OPENWEATHER_API_KEY

@st.cache_data(ttl=600, show_spinner=False)
def cached_get_live_weather_prediction(city_name: str):
    return get_live_weather_prediction(city_name)

@st.cache_data(ttl=600, show_spinner=False)
def cached_get_weather_forecast(city_name: str):
    return get_weather_forecast(city_name)

# -------------------------------------------------------------
# ALTAIR INTERACTIVE CHART GENERATORS WITH AXIS LEGENDS & LABELS  
# -------------------------------------------------------------

def render_rainfall_altair_chart(df_fc: pd.DataFrame):
    """Renders 24-Hour Rainfall Prediction Chart with explicit X & Y axis legends and tooltips."""
    if df_fc.empty or "rain_3h" not in df_fc.columns:
        st.info("Rainfall telemetry data unavailable.")
        return

    chart = alt.Chart(df_fc).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color='#0284C7').encode(
        x=alt.X('datetime:N', title='Date & Time (3-Hour Window)', sort=None, axis=alt.Axis(labelAngle=-30, labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        y=alt.Y('rain_3h:Q', title='Precipitation Volume (mm)', axis=alt.Axis(grid=True, labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        tooltip=[
            alt.Tooltip('datetime:N', title='Time Window'),
            alt.Tooltip('rain_3h:Q', title='Rainfall (mm)', format='.1f'),
            alt.Tooltip('weather_condition:N', title='Condition')
        ]
    ).properties(
        height=320
    ).configure_view(
        strokeWidth=0
    )
    st.altair_chart(chart, use_container_width=True)


def render_status_breakdown_chart(df_filtered: pd.DataFrame):
    """Renders Prediction Status Breakdown Chart with color keys, X-axis, and Y-axis legends."""
    if df_filtered.empty or "prediction" not in df_filtered.columns:
        st.info("Status data unavailable.")
        return

    status_counts = df_filtered["prediction"].value_counts().reset_index()
    status_counts.columns = ["Prediction Status", "Shipment Count"]

    chart = alt.Chart(status_counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('Prediction Status:N', title='Prediction Status Category', sort=['On Time', 'Delayed'], axis=alt.Axis(labelAngle=0, labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        y=alt.Y('Shipment Count:Q', title='Number of Shipments (Count)', axis=alt.Axis(grid=True, labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        color=alt.Color('Prediction Status:N', scale=alt.Scale(domain=['On Time', 'Delayed'], range=['#10B981', '#EF4444']), legend=alt.Legend(title="Status Legend", titleFontSize=11, labelFontSize=11)),
        tooltip=[
            alt.Tooltip('Prediction Status:N', title='Status'),
            alt.Tooltip('Shipment Count:Q', title='Total Shipments', format=',')
        ]
    ).properties(
        height=320
    ).configure_view(
        strokeWidth=0
    )
    st.altair_chart(chart, use_container_width=True)


def render_probability_distribution_chart(df_filtered: pd.DataFrame):
    """Renders Delay Probability Distribution Chart with area fill, 60% high-risk threshold line, and X & Y legends."""
    if "delay_probability" not in df_filtered.columns or df_filtered.empty:
        st.info("Delay probability data unavailable.")
        return

    prob_df = df_filtered[["delay_probability"]].sort_values("delay_probability", ascending=False).reset_index(drop=True).reset_index()
    prob_df.columns = ["Shipment Rank", "Delay Probability (%)"]
    prob_df["Shipment Rank"] += 1  # 1-based index

    # Gradient Area Chart
    area_chart = alt.Chart(prob_df).mark_area(
        line={'color': '#2563EB', 'size': 2},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#3B82F6', offset=0), alt.GradientStop(color='#EFF6FF', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Shipment Rank:Q', title='Shipments (Ranked by Risk Level: High → Low)', axis=alt.Axis(labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        y=alt.Y('Delay Probability (%):Q', title='Predicted Delay Risk Probability (%)', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=True, labelFontSize=11, titleFontSize=12, titleFontWeight='bold')),
        tooltip=[
            alt.Tooltip('Shipment Rank:Q', title='Shipment Rank #'),
            alt.Tooltip('Delay Probability (%):Q', title='Delay Risk', format='.1f')
        ]
    )

    # 60% High Risk Threshold Line
    threshold_df = pd.DataFrame({'Threshold': [60], 'Label': ['High Risk Cutoff (60%)']})
    rule = alt.Chart(threshold_df).mark_rule(color='#DC2626', strokeDash=[4, 4], size=2).encode(
        y='Threshold:Q'
    )

    combined_chart = (area_chart + rule).properties(
        height=320
    ).configure_view(
        strokeWidth=0
    )
    st.altair_chart(combined_chart, use_container_width=True)

# -------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Shipment Delay Predictor Pro",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM CSS DESIGN SYSTEM (ENTERPRISE GLASSMORPHISM & POPPINS THEME)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
        background-color: #F8FAFC;
    }

    /* Main Container Spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }

    /* Sidebar Navigation Radio Buttons Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.2rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #1E293B !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #0284C7 !important;
        background: #F0F9FF !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border: 1.5px solid #0284C7 !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.3) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #FFFFFF !important;
    }

    /* Hero Banner Header - Light Theme */
    .hero-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 50%, #E0F2FE 100%);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        color: #0F172A;
        box-shadow: 0 10px 30px -5px rgba(2, 132, 199, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #BAE6FD;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(255, 255, 255, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #0369A1;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0.4rem;
        font-weight: 500;
    }
    .accuracy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        color: #047857;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 700;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.12);
    }
    .pulse-dot {
        width: 10px;
        height: 10px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }

    /* Enterprise Container Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 20px !important;
        padding: 1.5rem 1.8rem !important;
        box-shadow: 0 8px 25px -5px rgba(15, 23, 42, 0.04), 0 3px 10px -2px rgba(15, 23, 42, 0.02) !important;
        margin-bottom: 1.6rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 14px 30px -5px rgba(15, 23, 42, 0.07) !important;
    }
    .card-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Weather Summary Metric Cards */
    .weather-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1.1rem;
        margin-top: 1rem;
    }
    .weather-metric-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .weather-metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(30, 58, 138, 0.08);
        border-color: #BFDBFE;
    }
    .weather-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-top: 0.2rem;
    }
    .weather-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .weather-condition-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.825rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    /* Prediction Button */
    .stButton > button[kind="primary"], div.stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    .predict-btn-wrap button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        height: 58px !important;
        width: 100% !important;
        border: none !important;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.25) !important;
    }
    .predict-btn-wrap button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.35) !important;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    }

    /* Status Result Cards */
    .status-ontime-card {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        color: #065F46;
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.12);
        border: 1px solid #A7F3D0;
    }
    .status-delayed-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        color: #991B1B;
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.12);
        border: 1px solid #FCA5A5;
    }
    .result-badge-text {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .result-subtext {
        font-size: 0.95rem;
        font-weight: 500;
        opacity: 0.95;
    }

    /* Analytics Badges Grid */
    .badges-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.2rem;
    }
    @media (max-width: 992px) {
        .badges-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 600px) {
        .badges-grid {
            grid-template-columns: 1fr;
        }
    }
    .analytics-badge-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.3rem;
        height: 100%;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .analytics-badge-item:hover {
        border-color: #CBD5E1;
        transform: translateY(-2px);
    }
    .badge-lbl {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-val {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* AI Insight Box */
    .ai-insight-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-left: 5px solid #0284C7;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.5rem;
    }
    .ai-insight-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0369A1;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }

    /* Input Card Container */
    .input-group-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.3rem;
        height: 100%;
    }
    .input-group-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Readout Pill */
    .readout-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        min-height: 68px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        box-sizing: border-box;
    }
    .readout-lbl {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        line-height: 1.2;
    }
    .readout-val {
        font-size: 0.88rem;
        color: #0F172A;
        font-weight: 700;
        text-align: right;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HERO HEADER SECTION
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div>
            <div class="hero-title">🚚 Shipment Delay Predictor Pro</div>
            <div class="hero-subtitle">AI Logistics Intelligence & Real-Time Weather Prediction</div>
        </div>
        <div class="accuracy-badge">
            <span class="pulse-dot"></span>
            Random Forest ML • 71.28% Accuracy
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "city_name" not in st.session_state:
    st.session_state["city_name"] = "Mumbai"
if "auto_temperature" not in st.session_state:
    st.session_state["auto_temperature"] = 28.5
if "auto_humidity" not in st.session_state:
    st.session_state["auto_humidity"] = 75
if "auto_wind_speed" not in st.session_state:
    st.session_state["auto_wind_speed"] = 14.0
if "auto_visibility" not in st.session_state:
    st.session_state["auto_visibility"] = 8.5
if "auto_rainfall" not in st.session_state:
    st.session_state["auto_rainfall"] = 0.0
if "auto_weather_condition" not in st.session_state:
    st.session_state["auto_weather_condition"] = "Cloudy"
if "weather_fetched" not in st.session_state:
    st.session_state["weather_fetched"] = False
if "weather_error" not in st.session_state:
    st.session_state["weather_error"] = None
if "has_predicted" not in st.session_state:
    st.session_state["has_predicted"] = False
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "last_feature_vector" not in st.session_state:
    st.session_state["last_feature_vector"] = {}
if "prediction_error" not in st.session_state:
    st.session_state["prediction_error"] = None
if "traffic_val" not in st.session_state:
    st.session_state["traffic_val"] = 25
if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = None
if "forecast_city" not in st.session_state:
    st.session_state["forecast_city"] = ""


def update_weather_state(city: str):
    """Fetches real-time weather telemetry for target hub and updates Streamlit session state."""
    try:
        w = cached_get_live_weather_prediction(city)
        st.session_state["city_name"] = w.get("city", city)
        st.session_state["auto_temperature"] = float(w.get("temperature") or 28.5)
        st.session_state["auto_humidity"] = int(float(w.get("humidity") or 75))
        st.session_state["auto_wind_speed"] = float(w.get("wind_speed") or 14.0)
        st.session_state["auto_visibility"] = float(w.get("visibility") or 8.5)
        st.session_state["auto_rainfall"] = float(w.get("rainfall") or 0.0)
        st.session_state["auto_weather_condition"] = w.get("weather_condition") or "Cloudy"
        st.session_state["weather_fetched"] = True
        st.session_state["weather_error"] = w.get("error")
        # Invalidate cached forecast so it is re-fetched for the new city
        st.session_state["forecast_data"] = None
        st.session_state["forecast_city"] = ""
        # KEY FIX: Reset prediction so it automatically re-runs with new weather data
        st.session_state["has_predicted"] = False
        st.session_state["prediction_result"] = None
        st.session_state["prediction_error"] = None
    except Exception as _wex:
        st.session_state["weather_error"] = f"Weather fetch failed: {_wex}"
        st.session_state["weather_fetched"] = False


def analyze_risk_factors(payload: dict) -> tuple:
    """Analyzes shipment attributes and weather telemetry to generate delay drivers and AI recommendations."""
    drivers = []
    recommendations = []

    w_cond = str(payload.get("weather_condition", "")).lower()
    rainfall = float(payload.get("rainfall", 0))
    visibility = float(payload.get("visibility", 10))
    humidity = float(payload.get("humidity", 50))
    dist = float(payload.get("distance_km", 0))
    traffic = int(payload.get("traffic_index", 0))
    maint = str(payload.get("maintenance_status", "")).lower()
    doc_ok = payload.get("documentation_complete", True)

    if w_cond == "storm" or rainfall >= 50:
        drivers.append("Heavy rainfall or storm conditions detected along transit corridor.")
        recommendations.append("Monitor live weather telemetry and consider delaying dispatch until weather clears.")
    elif w_cond == "rain" and rainfall > 10:
        drivers.append("Moderate rainfall slowing road surface speeds.")
        recommendations.append("Increase transit delivery buffer by +12 to +24 hours.")
    elif w_cond == "fog" or visibility < 3:
        drivers.append("Poor visibility (< 3 km) creating hazardous driving conditions.")
        recommendations.append("Ensure low-beam fog headlights and assign experienced drivers.")

    if humidity > 80:
        drivers.append("High relative humidity (> 80%) indicating potential precipitation risk.")

    if traffic >= 70:
        drivers.append("Severe traffic congestion detected along highway route (Traffic Index ≥ 70).")
        recommendations.append("Select alternate route or reschedule dispatch to off-peak hours.")
    elif traffic >= 45:
        drivers.append("Moderate traffic congestion along primary transport corridor.")

    if dist > 1000:
        drivers.append(f"Long transport distance ({dist:.0f} km) increases cumulative disruption exposure.")
        recommendations.append("Schedule driver relay swaps to avoid fatigue-related transit delays.")

    if maint == "under maintenance":
        drivers.append("Vehicle status currently marked as 'Under Maintenance'.")
        recommendations.append("Reassign shipment to a fully serviced vehicle in 'Good' condition.")
    elif maint == "due":
        drivers.append("Vehicle maintenance inspection is overdue.")
        recommendations.append("Perform quick fleet safety check prior to long-haul departure.")

    if not doc_ok:
        drivers.append("Customs / dispatch paperwork documentation is incomplete.")
        recommendations.append("Complete digital documentation paperwork to prevent terminal gate holds.")

    if not drivers:
        drivers.append("Optimal transit parameters: No major delay triggers detected.")
        recommendations.append("Proceed with standard dispatch schedule.")

    return drivers, recommendations


# -------------------------------------------------------------
# SIDEBAR NAVIGATION & ENTERPRISE METADATA
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Navigation Mode")
    app_mode = st.radio(
        "Select Application Mode",
        [
            "⚡ Delay Prediction Dashboard",
            "📁 Batch CSV Processor & Analytics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### ℹ️ Model Architecture")
        st.markdown("""
        - **Classifier**: Random Forest
        - **Accuracy**: **71.28%**
        - **Features**: 30 Predictors
        - **Engine**: Scikit-Learn Pipeline
        """)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### 📡 System Status")
        api_preview = f"{OPENWEATHER_API_KEY[:8]}..." if OPENWEATHER_API_KEY else "Connected"
        st.markdown(f"🟢 **Weather API**: `{api_preview}`")
        st.markdown("⚡ **Prediction Engine**: `Online`")

    with st.sidebar.expander("🔍 ML Model Input Features (Transparency)", expanded=False):
        st.caption("Exact 30 preprocessed feature inputs passed to `model.predict()`:")
        if st.session_state.get("last_feature_vector"):
            fv = st.session_state["last_feature_vector"]
            df_fv = pd.DataFrame(list(fv.items()), columns=["Feature Name", "Value Entered"])
            st.dataframe(df_fv, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("Click 'Calculate Shipment Delay Prediction' to view live preprocessed feature inputs.")

    st.markdown("---")
    st.caption("👨‍💻 *Logistics AI Intelligence System*")


# =============================================================
# MODE 1: MAIN DELAY PREDICTION DASHBOARD
# =============================================================
# MODE 1: MAIN DELAY PREDICTION DASHBOARD (DEFAULT)
# =============================================================
if "Batch CSV" not in str(app_mode):

    # ---------------------------------------------------------
    # SECTION 1: CARGO & PRIORITY
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="card-header-title">📦 Cargo & Priority</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            priority = st.selectbox("Priority Level 🔽", ["Standard", "Express", "Urgent"], index=0, help="Select shipment handling urgency priority.")
            shipment_type = st.selectbox("Shipment Type 🔽", ["Domestic", "Import", "Export"], index=0, help="Select logistics movement jurisdiction.")

        with col2:
            declared_value = st.number_input("Declared Value ($)", min_value=100.0, max_value=5000000.0, value=45000.0, step=1000.0, help="Declared commercial value of freight in USD.")
            weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=20000.0, value=250.0, step=10.0, help="Total gross weight in kilograms.")

        # Auto Weight Category calculation
        if weight_kg < 100:
            wt_cat = "Light (< 100 kg)"
            wt_badge = "🟢 Light Freight"
        elif weight_kg <= 500:
            wt_cat = "Medium (100 - 500 kg)"
            wt_badge = "🟡 Medium Freight"
        else:
            wt_cat = "Heavy (> 500 kg)"
            wt_badge = "🔴 Heavy Cargo"

        # Auto Value Density calculation
        val_density = declared_value / weight_kg if weight_kg > 0 else 0.0

        # Estimated Shipment Class calculation
        if declared_value > 100000 or priority == "Urgent":
            ship_class = "💎 High-Value Express"
        elif weight_kg > 1000:
            ship_class = "🚛 Bulk Heavy Transport"
        else:
            ship_class = "📦 Standard Logistics Freight"

        # Readout Pills Row (3 Balanced Columns)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown(f"""
            <div class="readout-pill">
                <span class="readout-lbl">Weight Category</span>
                <span class="readout-val">{wt_badge}</span>
            </div>
            """, unsafe_allow_html=True)
        with p_col2:
            st.markdown(f"""
            <div class="readout-pill">
                <span class="readout-lbl">Value Density</span>
                <span class="readout-val">₹{val_density:.2f} / kg</span>
            </div>
            """, unsafe_allow_html=True)
        with p_col3:
            st.markdown(f"""
            <div class="readout-pill">
                <span class="readout-lbl">Shipment Class</span>
                <span class="readout-val" style="color:#1E3A8A;">{ship_class}</span>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2: ROUTE & FLEET
    # ---------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="card-header-title">🚚 Route & Fleet</div>', unsafe_allow_html=True)

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            distance_km = st.number_input("Distance (km)", min_value=10.0, max_value=15000.0, value=380.0, step=25.0, help="Total travel distance between origin and destination hub.")
            route_risk = st.selectbox("Route Risk Level 🔽", ["Low", "Medium", "High"], index=0, help="Historical hazard risk level of the assigned highway corridor.")
            
            # Traffic slider with live readout
            traffic_index = st.slider(
                "Traffic Congestion Index",
                min_value=0,
                max_value=100,
                key="traffic_val",
                help="0 = Free Flowing Highway, 100 = Severe Urban Gridlock."
            )

        with r_col2:
            maintenance_status = st.selectbox("Vehicle Maintenance 🔽", ["Good", "Due", "Under Maintenance"], index=0, help="Fleet vehicle mechanical inspection status.")
            expected_transit_days = st.number_input(
                "Expected Transit Days ⏱️",
                min_value=0.5,
                max_value=30.0,
                value=1.0,
                step=0.5,
                help="Enter expected transit days directly (e.g., 1 day, 2 days, 5 days)."
            )
            st.write("")
            doc_complete_toggle = st.toggle("Documentation Complete", value=True, help="Toggle whether customs and dispatch paperwork is fully verified.")
            documentation_complete = bool(doc_complete_toggle)

        ship_date = datetime.now().date()
        expected_delivery_date = ship_date + timedelta(days=float(expected_transit_days))

        # Auto Route Difficulty calculation
        if traffic_index >= 75 or (traffic_index >= 50 and route_risk == "High"):
            diff_label = "🔴 Severe Gridlock / High Risk"
        elif traffic_index >= 45 or route_risk == "High":
            diff_label = "🟠 High Route Difficulty"
        elif traffic_index >= 20 or route_risk == "Medium":
            diff_label = "🟡 Moderate Transit Effort"
        else:
            diff_label = "🟢 Low Difficulty (Optimal)"

        # Readout Pills Row (2 Balanced Columns)
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        rp_col1, rp_col2 = st.columns(2)
        with rp_col1:
            st.markdown(f"""
            <div class="readout-pill">
                <span class="readout-lbl">Route Difficulty</span>
                <span class="readout-val">{diff_label}</span>
            </div>
            """, unsafe_allow_html=True)
        with rp_col2:
            st.markdown(f"""
            <div class="readout-pill">
                <span class="readout-lbl">Target Delivery</span>
                <span class="readout-val">📅 {expected_delivery_date.strftime('%b %d, %Y')}</span>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-header-title">🌦 Live Weather</div>', unsafe_allow_html=True)

        st.write("Select an **Indian Metropolitan Logistics Hub** or query a custom location to fetch real-time atmospheric telemetry:")

        # Quick Access Indian Metropolitan Hubs
        indian_hubs = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Pune", "Surat", "Jaipur", "Lucknow", "Nagpur"]
        
        hub_cols = st.columns(6)
        for idx, hub in enumerate(indian_hubs):
            with hub_cols[idx % 6]:
                if st.button(f"🏙️ {hub}", key=f"indian_hub_{hub}", use_container_width=True):
                    update_weather_state(hub)
                    st.rerun()

        st.write("")
        w_input_col, w_btn_col = st.columns([3.5, 1.5], vertical_alignment="bottom")
        with w_input_col:
            city_input = st.text_input(
                "Search Custom City",
                value=st.session_state["city_name"],
                placeholder="e.g. Nagpur, Indore, Chandigarh, London, Tokyo...",
                help="Query any city globally via OpenWeatherMap API."
            )

        with w_btn_col:
            if st.button("🌤️ Fetch Live Weather", type="primary", use_container_width=True):
                update_weather_state(city_input)
                st.rerun()

        # Weather Condition Badge formatting
        cond_style = {
            "Clear": ("🟢 Clear", "#DEF7EC", "#03543F"),
            "Cloudy": ("🟡 Cloudy", "#FEF08A", "#854D0E"),
            "Rain": ("🔵 Rain", "#DBEAFE", "#1E40AF"),
            "Storm": ("🔴 Storm", "#FEE2E2", "#991B1B"),
            "Fog": ("🌫️ Fog", "#F1F5F9", "#475569"),
            "Snow": ("❄️ Snow", "#E0F2FE", "#0369A1")
        }
        cur_cond = st.session_state["auto_weather_condition"]
        cond_text, cond_bg, cond_fg = cond_style.get(cur_cond, ("🟢 Clear", "#DEF7EC", "#03543F"))

        # Weather Summary Cards
        st.markdown(f"""
        <div style="margin-top: 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;">
            <span style="font-weight: 700; color: #1E3A8A; font-size: 1.05rem;">
                Destination Telemetry: <b>{st.session_state['city_name']}</b>
            </span>
            <span class="weather-condition-badge" style="background: {cond_bg}; color: {cond_fg};">
                {cond_text}
            </span>
        </div>

        <div class="weather-metrics-grid">
            <div class="weather-metric-box">
                <div style="font-size: 1.5rem;">🌡️</div>
                <div class="weather-val">{st.session_state['auto_temperature']} °C</div>
                <div class="weather-lbl">Temperature</div>
            </div>
            <div class="weather-metric-box">
                <div style="font-size: 1.5rem;">💧</div>
                <div class="weather-val">{st.session_state['auto_humidity']}%</div>
                <div class="weather-lbl">Humidity</div>
            </div>
            <div class="weather-metric-box">
                <div style="font-size: 1.5rem;">🌧️</div>
                <div class="weather-val">{st.session_state['auto_rainfall']} mm</div>
                <div class="weather-lbl">Rainfall</div>
            </div>
            <div class="weather-metric-box">
                <div style="font-size: 1.5rem;">💨</div>
                <div class="weather-val">{st.session_state['auto_wind_speed']} km/h</div>
                <div class="weather-lbl">Wind Speed</div>
            </div>
            <div class="weather-metric-box">
                <div style="font-size: 1.5rem;">👁️</div>
                <div class="weather-val">{st.session_state['auto_visibility']} km</div>
                <div class="weather-lbl">Visibility</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("weather_error"):
            st.info(f"ℹ️ Weather Telemetry Note: {st.session_state['weather_error']}")

    # ---------------------------------------------------------
    # SECTION 4: AI PREDICTION BUTTON
    # ---------------------------------------------------------
    predict_clicked = st.button("🚀 Calculate Shipment Delay Prediction", type="primary", use_container_width=True)

    # Build input payload from current widget values
    input_payload = {
        "priority": priority,
        "weight_kg": weight_kg,
        "declared_value": declared_value,
        "shipment_type": shipment_type,
        "distance_km": distance_km,
        "traffic_index": traffic_index,
        "route_risk": route_risk,
        "maintenance_status": maintenance_status,
        "documentation_complete": documentation_complete,
        "ship_date": ship_date.strftime("%Y-%m-%d"),
        "expected_delivery_date": expected_delivery_date.strftime("%Y-%m-%d"),
        "Expected Transit Days": expected_transit_days,
        "weather_condition": st.session_state["auto_weather_condition"],
        "temperature": st.session_state["auto_temperature"],
        "humidity": st.session_state["auto_humidity"],
        "visibility": st.session_state["auto_visibility"],
        "rainfall": st.session_state["auto_rainfall"]
    }

    # -----------------------------------------------------------------------
    # ROOT-CAUSE FIX: Run prediction ONLY on button click (or first-time load).
    # Previously, predict_delay() ran unconditionally on every Streamlit rerun
    # (every widget change), blocking the event loop for ~100ms per call with
    # a 200-tree Random Forest + ensemble agreement loop, causing Streamlit to
    # display a black overlay and appear frozen/crashed.
    # Solution: Store result in session_state; reuse cached result on reruns.
    # -----------------------------------------------------------------------
    if predict_clicked or not st.session_state.get("has_predicted"):
        with st.spinner("🔄 Running Random Forest inference..."):
            try:
                res = predict_delay(input_payload)
                if not isinstance(res, dict):
                    raise ValueError(f"predict_delay returned unexpected type: {type(res)}")
                # Store successful result
                st.session_state["prediction_result"] = res
                st.session_state["last_feature_vector"] = res.get("feature_vector") or {}
                st.session_state["has_predicted"] = True
                st.session_state["prediction_error"] = None
            except Exception as _pred_exc:
                import traceback as _tb
                st.session_state["prediction_error"] = _tb.format_exc()
                st.session_state["prediction_result"] = {
                    "prediction": "Error",
                    "delay_probability": 0.0,
                    "confidence": 0.0,
                    "expected_transit_days": 0.0,
                    "expected_delivery_date": "N/A",
                    "prediction_class": -1,
                    "feature_vector": {},
                    "error": str(_pred_exc)
                }

    # Retrieve the stored result (never None after first prediction)
    res = st.session_state.get("prediction_result")
    if res is None:
        # Guard: show placeholder result until first prediction is run
        res = {
            "prediction": "On Time",
            "delay_probability": 0.0,
            "confidence": 0.0,
            "expected_transit_days": float(expected_transit_days),
            "expected_delivery_date": expected_delivery_date.strftime("%Y-%m-%d"),
            "prediction_class": 0,
            "feature_vector": {},
            "error": None
        }

    # Show any unhandled prediction errors prominently (does not crash the app)
    if st.session_state.get("prediction_error"):
        with st.expander("⚠️ Prediction Error Details (click to expand)", expanded=True):
            st.error("An error occurred during prediction. The app has not crashed — details below:")
            st.code(st.session_state["prediction_error"], language="python")

    # ---------------------------------------------------------
    # SECTION 5: PREDICTION RESULT & AI INSIGHTS
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-header-title">📊 Prediction Result & Risk Assessment</div>', unsafe_allow_html=True)

        if not st.session_state.get("has_predicted"):
            st.info("👆 Click **Calculate Shipment Delay Prediction** above to run the AI model and see your results here.")
        else:
            _prediction_label = res.get("prediction", "On Time")
            _delay_prob = float(res.get("delay_probability") or 0.0)
            _confidence = float(res.get("confidence") or 0.0)

        res_col1, res_col2, res_col3 = st.columns([1.3, 1, 1], vertical_alignment="center")

        with res_col1:
            if res.get("prediction", "On Time") == "On Time":
                st.markdown("""
                <div class="status-ontime-card">
                    <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;">PREDICTION STATUS</div>
                    <div class="result-badge-text">✅ ON TIME</div>
                    <div class="result-subtext">Low disruption probability. Transit parameters within nominal operating bounds.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-delayed-card">
                    <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;">PREDICTION STATUS</div>
                    <div class="result-badge-text">❌ DELAYED</div>
                    <div class="result-subtext">Elevated delay risk detected along scheduled transport corridor.</div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            _delay_prob = float(res.get("delay_probability") or 0.0)
            st.metric("Delay Probability", f"{_delay_prob}%")
            st.progress(min(1.0, max(0.0, _delay_prob / 100.0)))
            risk_lvl = "🟢 Low Risk" if _delay_prob < 35 else ("🟡 Moderate Risk" if _delay_prob < 65 else "🔴 High Delay Risk")
            st.caption(f"Risk Rating: **{risk_lvl}**")

        with res_col3:
            _confidence = float(res.get("confidence") or 0.0)
            conf_badge = "🟢 High" if _confidence >= 75 else ("🟡 Medium" if _confidence >= 60 else "🔴 Low")
            st.metric("Model Confidence", f"{_confidence}%")
            st.progress(min(1.0, max(0.0, _confidence / 100.0)))
            st.caption(f"Confidence Level: **{conf_badge}**")

        # Analytics Badges Grid
        st.markdown("<h4 style='font-size:1.05rem; font-weight:700; margin-top:1.5rem; color:#1E293B;'>Operational Analytics Badges</h4>", unsafe_allow_html=True)

        _delay_prob_badge = float(res.get("delay_probability") or 0.0)
        w_sev = "🔴 Severe Hazard" if st.session_state["auto_weather_condition"] in ["Storm", "Snow"] or float(st.session_state.get("auto_rainfall") or 0) > 50 else ("🟡 Moderate Impact" if st.session_state["auto_weather_condition"] in ["Rain", "Fog"] else "🟢 Minimal Impact")
        t_sev = "🔴 Heavy Gridlock" if traffic_index >= 75 else ("🟡 Congested Corridor" if traffic_index >= 40 else "🟢 Fluid Traffic")
        r_score = "🔴 High Risk Score" if _delay_prob_badge >= 60 else ("🟡 Medium Risk Score" if _delay_prob_badge >= 30 else "🟢 Low Risk Score")
        t_score = "⚡ Expedited Transit" if expected_transit_days <= 1.5 else ("⏱ Standard Transit" if expected_transit_days <= 4.0 else "🐢 Extended Transit")
        d_cat = "Short (< 300 km)" if distance_km < 300 else ("Regional (300 - 1000 km)" if distance_km <= 1000 else "Long-Haul (> 1000 km)")
        doc_stat = "✅ Verified Complete" if documentation_complete else "❌ Incomplete / Pending"
        maint_stat = "🟢 Operational / Good" if maintenance_status == "Good" else ("🟡 Inspection Overdue" if maintenance_status == "Due" else "🔴 Under Maintenance")

        st.markdown(f"""
        <div class="badges-grid">
            <div class="analytics-badge-item">
                <span class="badge-lbl">Weather Severity</span>
                <span class="badge-val">{w_sev}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Traffic Severity</span>
                <span class="badge-val">{t_sev}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Risk Score</span>
                <span class="badge-val">{r_score}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Transit Score</span>
                <span class="badge-val">{t_score}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Distance Category</span>
                <span class="badge-val">{d_cat}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Weight Category</span>
                <span class="badge-val">{wt_cat}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Documentation Status</span>
                <span class="badge-val">{doc_stat}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">Maintenance Status</span>
                <span class="badge-val">{maint_stat}</span>
            </div>
            <div class="analytics-badge-item">
                <span class="badge-lbl">User Target Delivery</span>
                <span class="badge-val">📅 {expected_delivery_date.strftime('%b %d, %Y')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Shipment Analysis & Recommendations
        delay_drivers, ai_recs = analyze_risk_factors(input_payload)

        drivers_html = "".join([f"<li>{d}</li>" for d in delay_drivers])
        recs_html = "".join([f"<li>{r}</li>" for r in ai_recs])

        st.markdown(
            f'<div class="ai-insight-card">'
            f'<div class="ai-insight-header">🤖 AI Shipment Analysis & Recommendations</div>'
            f'<div style="margin-bottom: 0.8rem;">'
            f'<b>Identified Delay Risk Drivers:</b>'
            f'<ul style="margin-top: 0.3rem; padding-left: 1.2rem; color: #1E293B;">{drivers_html}</ul>'
            f'</div>'
            f'<div>'
            f'<b>Actionable Supply Chain Recommendations:</b>'
            f'<ul style="margin-top: 0.3rem; padding-left: 1.2rem; color: #0369A1;">{recs_html}</ul>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # SECTION 6: WEATHER FORECAST PANELS
    # ---------------------------------------------------------
    with st.container(border=True):
        _forecast_city = st.session_state.get("city_name", "Mumbai")
        st.markdown(f'<div class="card-header-title">📈 Weather Forecast Telemetry ({_forecast_city})</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🕒 Hourly Forecast", "🌧️ 24-Hour Rainfall Prediction", "📅 5-Day Forecast"])

        # Only fetch forecast if city changed or not yet fetched — prevents crash on every rerun
        if st.session_state.get("forecast_city") != _forecast_city or st.session_state.get("forecast_data") is None:
            try:
                _fc = cached_get_weather_forecast(_forecast_city)
                st.session_state["forecast_data"] = _fc
                st.session_state["forecast_city"] = _forecast_city
            except Exception as _fcex:
                st.session_state["forecast_data"] = {"success": False, "entries": [], "error": str(_fcex)}
                st.session_state["forecast_city"] = _forecast_city

        fc_data = st.session_state.get("forecast_data") or {"success": False, "entries": []}

        with tab1:
            if fc_data.get("success") and fc_data.get("entries"):
                try:
                    df_fc = pd.DataFrame(fc_data["entries"])
                    _hourly_cols = [c for c in ["datetime", "weather_condition", "description", "temp", "rain_pop", "rain_3h", "wind_speed"] if c in df_fc.columns]
                    df_hourly = df_fc.head(8)[_hourly_cols]
                    df_hourly.columns = ["Date & Time", "Condition", "Details", "Temp (°C)", "Rain Prob (%)", "3h Rain (mm)", "Wind (km/h)"][:len(_hourly_cols)]
                    st.dataframe(df_hourly, use_container_width=True, hide_index=True)
                except Exception as _e:
                    st.warning(f"Hourly forecast display error: {_e}")
            else:
                st.warning("Hourly forecast telemetry currently unavailable.")

        with tab2:
            if fc_data.get("success") and fc_data.get("entries"):
                try:
                    df_fc = pd.DataFrame(fc_data["entries"]).head(8)
                    render_rainfall_altair_chart(df_fc)
                except Exception as _e:
                    st.info(f"24-Hour rainfall chart error: {_e}")
            else:
                st.info("24-Hour rainfall prediction chart unavailable.")

        with tab3:
            if fc_data.get("success") and fc_data.get("entries"):
                try:
                    df_fc = pd.DataFrame(fc_data["entries"])
                    _5d_cols = [c for c in ["datetime", "weather_condition", "description", "temp", "humidity", "wind_speed"] if c in df_fc.columns]
                    df_5d = df_fc[_5d_cols]
                    df_5d.columns = ["Date & Time", "Condition", "Description", "Temp (°C)", "Humidity (%)", "Wind Speed (km/h)"][:len(_5d_cols)]
                    st.dataframe(df_5d, use_container_width=True, hide_index=True)
                except Exception as _e:
                    st.warning(f"5-Day forecast display error: {_e}")
            else:
                st.warning("5-Day forecast telemetry currently unavailable.")


# =============================================================
# MODE 2: BATCH CSV PROCESSOR & ADVANCED ANALYTICS
# =============================================================
else:
    with st.container(border=True):
        st.markdown('<div class="card-header-title">📁 Bulk Batch CSV Processor & Advanced Analytics</div>', unsafe_allow_html=True)
        st.write("Upload a CSV dataset of shipments or load sample telemetry to execute bulk machine learning inference, filter predictions, and explore risk analytics.")

        # Case-insensitive fallback: 'Data' on Windows, 'data' on Linux (Streamlit Cloud)
        _data_dir = os.path.join(PROJECT_ROOT, "Data")
        if not os.path.exists(_data_dir):
            _data_dir = os.path.join(PROJECT_ROOT, "data")
        sample_batch_file = os.path.join(_data_dir, "sample_batch.csv")

        c1, c2, c3 = st.columns([1, 1.2, 2])
        with c1:
            if os.path.exists(sample_batch_file):
                with open(sample_batch_file, "rb") as f:
                    st.download_button("📥 Download Sample CSV", f, "sample_batch.csv", "text/csv", use_container_width=True)

        with c2:
            if os.path.exists(sample_batch_file):
                if st.button("⚡ Load Sample Dataset", type="secondary", use_container_width=True):
                    df_in = pd.read_csv(sample_batch_file)
                    st.session_state["raw_upload_df"] = df_in
                    with st.spinner("Processing sample batch predictions..."):
                        df_out = predict_batch(df_in)
                        st.session_state["batch_processed_df"] = df_out

        with c3:
            csv_upload = st.file_uploader("Upload Custom CSV Batch File", type=["csv"], label_visibility="collapsed")

        if csv_upload is not None:
            try:
                df_in = pd.read_csv(csv_upload)
                st.session_state["raw_upload_df"] = df_in
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        if st.session_state.get("raw_upload_df") is not None:
            df_in = st.session_state["raw_upload_df"]
            st.markdown(f"**Loaded Dataset Preview** (`{len(df_in)}` records):")
            st.dataframe(df_in.head(4), use_container_width=True)

            if st.button("⚙️ Execute Bulk Predictions", type="primary", use_container_width=True):
                with st.spinner("Processing predictions with Random Forest model..."):
                    df_out = predict_batch(df_in)
                    st.session_state["batch_processed_df"] = df_out

    # Output & Interactive Filter Section
    if st.session_state.get("batch_processed_df") is not None:
        df_out = st.session_state["batch_processed_df"]

        with st.container(border=True):
            st.markdown('<div class="card-header-title">🎛️ Interactive Filters & Data Exploration</div>', unsafe_allow_html=True)
            
            f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1.2, 1.5], vertical_alignment="bottom")

            with f_col1:
                status_options = ["All", "Delayed", "On Time"]
                selected_status = st.selectbox("Filter Status 🔽", status_options, index=0)

            with f_col2:
                priority_options = ["All"]
                if "priority" in df_out.columns:
                    priority_options += sorted([str(x) for x in df_out["priority"].dropna().unique().tolist()])
                selected_priority = st.selectbox("Filter Priority 🔽", priority_options, index=0)

            with f_col3:
                min_prob, max_prob = st.slider(
                    "Delay Prob Range (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 100.0),
                    step=5.0
                )

            with f_col4:
                search_query = st.text_input("Search Dataset 🔍", placeholder="Search city, priority, status...", help="Global search across all text columns.")

            # Filter Logic
            df_filtered = df_out.copy()

            if selected_status != "All":
                df_filtered = df_filtered[df_filtered["prediction"] == selected_status]

            if selected_priority != "All" and "priority" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["priority"].astype(str) == selected_priority]

            if "delay_probability" in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered["delay_probability"] >= min_prob) & 
                    (df_filtered["delay_probability"] <= max_prob)
                ]

            if search_query.strip():
                query_str = search_query.strip().lower()
                mask = df_filtered.apply(
                    lambda row: row.astype(str).str.lower().str.contains(query_str).any(), axis=1
                )
                df_filtered = df_filtered[mask]

            # Dynamic Metrics Update
            total_records = len(df_filtered)
            cnt_delayed = int((df_filtered["prediction"] == "Delayed").sum()) if total_records > 0 else 0
            cnt_ontime = int((df_filtered["prediction"] == "On Time").sum()) if total_records > 0 else 0
            cnt_high_risk = int((df_filtered["delay_probability"] >= 60.0).sum()) if (total_records > 0 and "delay_probability" in df_filtered.columns) else 0
            avg_prob = round(float(df_filtered["delay_probability"].mean()), 2) if (total_records > 0 and "delay_probability" in df_filtered.columns) else 0.0

            st.markdown("---")
            st.markdown("### 📊 Live Key Performance Indicators (KPIs)")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Filtered Records", f"{total_records} / {len(df_out)}")
            k2.metric("On-Time", f"{cnt_ontime}", delta=f"{round(cnt_ontime/total_records*100,1)}%" if total_records > 0 else "0%")
            k3.metric("Delayed", f"{cnt_delayed}", delta=f"-{round(cnt_delayed/total_records*100,1)}%" if total_records > 0 else "0%", delta_color="inverse")
            k4.metric("High Risk (≥60%)", f"{cnt_high_risk}", delta_color="inverse")
            k5.metric("Avg Delay Prob", f"{avg_prob}%")

            # Dynamic Charts Grid
            if total_records > 0:
                st.markdown("---")
                st.markdown("### 📈 Batch Risk Visualizations")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("#### 🎯 Prediction Status Breakdown")
                    render_status_breakdown_chart(df_filtered)

                with chart_col2:
                    st.markdown("#### ⚡ Delay Probability Distribution")
                    render_probability_distribution_chart(df_filtered)

            # Table & Export Buttons
            st.markdown("---")
            st.markdown(f"### 📋 Processed Batch Dataset (`{total_records}` records)")
            st.dataframe(df_filtered, use_container_width=True)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                out_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Full Batch CSV", out_bytes, "predicted_full_batch.csv", "text/csv", type="primary", use_container_width=True)

            with d_col2:
                filtered_bytes = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("🔍 Download Filtered Dataset CSV", filtered_bytes, "predicted_filtered_batch.csv", "text/csv", type="secondary", use_container_width=True)
