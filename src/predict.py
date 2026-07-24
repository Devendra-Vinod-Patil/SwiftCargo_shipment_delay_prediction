import os
import sys
import joblib
import pandas as pd
from datetime import datetime, timedelta
import math

# Path resolution to load artifacts from models/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Add project root to sys.path if not present
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_single_record, preprocess_batch

# Look for model files in models/ directory or root directory fallback
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = PROJECT_ROOT

MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SELECTED_FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load model and selected features
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
except Exception as e:
    model = None
    selected_features = []
    model_load_error = f"Model load error: {str(e)}"

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None


def compute_transit_and_delivery_date(rec: dict):
    """
    Computes/predicts expected transit days and expected delivery date
    based on raw input dates, explicit user inputs, or distance.
    """
    if not isinstance(rec, dict):
        rec = {}

    ship_date = pd.to_datetime(rec.get('ship_date'), errors='coerce')
    exp_delivery_date = pd.to_datetime(rec.get('expected_delivery_date'), errors='coerce')

    if pd.notnull(exp_delivery_date) and pd.notnull(ship_date):
        days = max(0.5, float((exp_delivery_date - ship_date).days))
        deliv_str = exp_delivery_date.strftime('%Y-%m-%d')
    elif rec.get('Expected Transit Days') is not None:
        days = float(rec['Expected Transit Days'])
        base_dt = ship_date if pd.notnull(ship_date) else datetime.now()
        deliv_str = (base_dt + timedelta(days=days)).strftime('%Y-%m-%d')
    elif rec.get('expected_transit_days') is not None:
        days = float(rec['expected_transit_days'])
        base_dt = ship_date if pd.notnull(ship_date) else datetime.now()
        deliv_str = (base_dt + timedelta(days=days)).strftime('%Y-%m-%d')
    else:
        dist_val = float(rec.get('distance_km', 350.0))
        days = max(1.0, float(round(dist_val / 350.0, 1)))
        base_dt = ship_date if pd.notnull(ship_date) else datetime.now()
        deliv_str = (base_dt + timedelta(days=days)).strftime('%Y-%m-%d')

    return days, deliv_str


def predict_delay(input_data):
    """
    Predicts shipment delay based on input data (dictionary).
    Supports raw input records or preprocessed feature dictionaries.
    Returns prediction status, probability, confidence, expected transit days & delivery date.
    """
    if not isinstance(input_data, dict) or not input_data:
        return {
            "prediction": "Error",
            "delay_probability": 0.0,
            "confidence": 0.0,
            "expected_transit_days": 0.0,
            "expected_delivery_date": "N/A",
            "prediction_class": -1,
            "error": "Input must be a non-empty dictionary."
        }

    if model is None or len(selected_features) == 0:
        return {
            "prediction": "Error",
            "delay_probability": 0.0,
            "confidence": 0.0,
            "expected_transit_days": 0.0,
            "expected_delivery_date": "N/A",
            "prediction_class": -1,
            "error": f"Model artifacts could not be loaded. Details: {model_load_error}"
        }

    try:
        # Preprocess record into feature dataframe
        df = preprocess_single_record(input_data, selected_features)

        # Compute expected transit days & delivery date
        transit_days, exp_deliv_date = compute_transit_and_delivery_date(input_data)

        # Base ML Model Probability
        raw_prob = float(model.predict_proba(df)[0][1])
        raw_prob_clamped = max(0.01, min(0.99, raw_prob))

        # Convert base ML probability to Log-Odds (Logit) space as anchor
        base_logit = math.log(raw_prob_clamped / (1.0 - raw_prob_clamped))

        # 1. Schedule Tightness Risk (Log-Odds shift)
        dist_km = float(input_data.get('distance_km', df['distance_km'].values[0] if 'distance_km' in df else 350.0))
        rec_min_days = max(1.0, float(round(dist_km / 350.0, 1)))

        schedule_shift = 0.0
        if transit_days < rec_min_days:
            gap_ratio = rec_min_days / max(0.5, transit_days)
            schedule_shift = math.log(gap_ratio) * 1.65

        # 2. Weather Telemetry Risk (Log-Odds shift)
        weather_cond = str(input_data.get('weather_condition', '')).title()
        rainfall_mm = float(input_data.get('rainfall', 0.0))
        visibility_km = float(input_data.get('visibility', 10.0))
        wind_speed_kmh = float(input_data.get('wind_speed', 0.0))

        weather_shift = 0.0
        if weather_cond in ["Storm", "Snow"] or rainfall_mm >= 30.0:
            weather_shift = 1.2
        elif weather_cond in ["Rain", "Fog"] or rainfall_mm >= 10.0 or visibility_km < 3.0:
            weather_shift = 0.6

        if wind_speed_kmh > 40.0:
            weather_shift += 0.3

        # 3. Priority Risk (Log-Odds shift)
        # Ensure higher priority always monotonically lowers the probability of delay
        p_in = input_data.get('priority')
        if isinstance(p_in, str):
            p_str = p_in.lower()
            if p_str == 'urgent': priority_val = 3.0
            elif p_str == 'express': priority_val = 2.0
            else: priority_val = 1.0
        elif p_in is not None:
            try: priority_val = float(p_in)
            except: priority_val = 1.0
        else:
            priority_val = float(df['priority'].values[0] if 'priority' in df else 1.0)
            
        priority_shift = 0.0
        if priority_val >= 3.0:
            priority_shift = -0.65  # Urgent significantly reduces delay risk
        elif priority_val == 2.0:
            priority_shift = -0.25  # Express moderately reduces risk
        else:
            priority_shift = +0.25  # Standard increases risk

        # Combine Log-Odds and transform back to probability via Sigmoidal function
        total_logit = base_logit + schedule_shift + weather_shift + priority_shift
        calibrated_prob = 1.0 / (1.0 + math.exp(-total_logit))

        # Final prediction and distinct Model Confidence (Ensemble Tree Agreement & Certainty)
        final_prediction = 1 if calibrated_prob >= 0.50 else 0
        final_status = "Delayed" if final_prediction == 1 else "On Time"

        try:
            tree_preds = [float(t.predict(df.values)[0]) for t in model.estimators_]
            target_class = float(final_prediction)
            ensemble_tree_agreement = sum(1 for tp in tree_preds if tp == target_class) / len(tree_preds)
        except Exception:
            ensemble_tree_agreement = max(raw_prob_clamped, 1.0 - raw_prob_clamped)

        certainty_margin = 0.50 + (abs(calibrated_prob - 0.50) * 0.85)
        final_confidence = float(round(max(ensemble_tree_agreement, certainty_margin) * 100, 2))

        # Extract preprocessed feature vector dictionary for full model transparency
        feature_vector = df.iloc[0].to_dict()

        result = {
            "prediction": final_status,
            "delay_probability": float(round(calibrated_prob * 100, 2)),
            "confidence": final_confidence,
            "expected_transit_days": transit_days,
            "expected_delivery_date": exp_deliv_date,
            "prediction_class": final_prediction,
            "feature_vector": feature_vector,
            "error": None
        }
        return result
    except Exception as e:
        return {
            "prediction": "Error",
            "delay_probability": 0.0,
            "confidence": 0.0,
            "expected_transit_days": 0.0,
            "expected_delivery_date": "N/A",
            "prediction_class": -1,
            "error": str(e)
        }


def predict_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts shipment delay for a batch DataFrame of shipment records.
    Returns the DataFrame with appended prediction and auto-predicted transit columns.
    """
    if model is None or len(selected_features) == 0:
        df_res = df_batch.copy()
        df_res["prediction"] = "Error"
        df_res["delay_probability"] = 0.0
        df_res["confidence"] = 0.0
        df_res["expected_transit_days"] = 0.0
        df_res["expected_delivery_date"] = "N/A"
        return df_res

    # Use predict_delay to ensure all logit shifts (weather, schedule, priority) are applied identically
    records = df_batch.to_dict(orient="records")
    results = [predict_delay(r) for r in records]

    df_res = df_batch.copy()
    df_res["expected_transit_days"] = [r.get("expected_transit_days", 0.0) for r in results]
    df_res["expected_delivery_date"] = [r.get("expected_delivery_date", "N/A") for r in results]
    df_res["prediction_class"] = [r.get("prediction_class", -1) for r in results]
    df_res["prediction"] = [r.get("prediction", "Error") for r in results]
    df_res["delay_probability"] = [r.get("delay_probability", 0.0) for r in results]
    df_res["confidence"] = [r.get("confidence", 0.0) for r in results]

    return df_res


if __name__ == "__main__":
    sample_normal = {
        "priority": "Standard",
        "distance_km": 800,
        "traffic_index": 20,
        "weather_condition": "Clear"
    }

    print("=" * 60)
    print(" SHIPMENT DELAY PREDICTION ENGINE (src/predict.py)")
    print("=" * 60)
    res = predict_delay(sample_normal)
    print("Prediction Result:", res)
    print("=" * 60)
