# 🚚 Shipment Delay Prediction System — Project Documentation (`project.md`)

This document provides a complete technical explanation of the **Shipment Delay Prediction Architecture**, detailing the exact project execution flow, model `.pkl` artifact integration, proof of deterministic model inference, the diagnosis and resolution of the previous prediction issue, and future feature enhancement recommendations.

---

## 1. Exact Project Execution Flow

The end-to-end flow of a prediction request through the system is structured as follows:

```text
[ User Input / Raw Dictionary / CSV Batch ]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. API Entry Point (src/predict.py)                         │
│    - Accepts raw or partial Python dictionary / DataFrame   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Preprocessing Engine (src/preprocess.py)                 │
│    - Date Calculations: dispatch_lead_time & Expected Transit│
│    - Derived Ratios: vehicle_utilization & value_density    │
│    - Categorical Encoding: priority, maintenance, risk, etc.│
│    - Imputation: Fills missing fields with DATASET MEDIANS   │
│    - Alignment: Generates 1x30 array for selected_features  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Model Loading & Inference (models/random_forest_model.pkl)│
│    - Passes 30 features into loaded RandomForestClassifier  │
│    - Evaluates 100 Decision Trees                           │
│    - Computes class (0: On Time, 1: Delayed) & probability  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Output Formatting & Dashboard Rendering                   │
│    - Returns prediction status, delay_probability %,        │
│      confidence %, and prediction_class                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Does It Use Our PKL Files or Is It Random? (Proof of Non-Random Inference)

> [!IMPORTANT]
> **Predictions are 100% deterministic, non-random, and directly computed using the saved model artifacts (`.pkl` files) loaded from the `models/` directory.**

### Proof of PKL Artifact Usage

The system directly loads and uses three trained `.pkl` artifacts:

1. **`models/random_forest_model.pkl`**:
   - A trained `RandomForestClassifier` model (100 decision trees) trained on historical shipment data.
   - Loaded via `joblib.load("models/random_forest_model.pkl")`.
2. **`models/selected_features.pkl`**:
   - A pickled Python list containing the **exact 30 feature names** selected via Recursive Feature Elimination (RFECV).
   - Loaded via `joblib.load("models/selected_features.pkl")`.
3. **`models/scaler.pkl`**:
   - A fitted `StandardScaler` artifact saved during hyperparameter experimentation.

### Code Proof of Model Inference

In [src/predict.py](file:///c:/technetic_internship/src/predict.py):

```python
import joblib

# 1. Load trained model & feature list directly from PKL files
model = joblib.load("models/random_forest_model.pkl")
selected_features = joblib.load("selected_features.pkl")

def predict_delay(input_data):
    # 2. Preprocess input into 1x30 feature DataFrame matching selected_features
    df_features = preprocess_single_record(input_data, selected_features)
    
    # 3. Model Inference (Non-Random)
    prediction = model.predict(df_features)[0]          # Returns 0 (On Time) or 1 (Delayed)
    probability = model.predict_proba(df_features)[0][1] # Computes exact ensemble tree vote ratio
    
    return {
        "prediction": "Delayed" if prediction == 1 else "On Time",
        "delay_probability": float(round(probability * 100, 2)),
        "confidence": float(round(max(probability, 1 - probability) * 100, 2)),
        "prediction_class": int(prediction)
    }
```

### Determinism Verification
If you run `predict_delay(input_data)` 1,000 times on the exact same input dictionary, it will return the **exact same delay probability (e.g., `24.97%`) every single time**. It contains zero random number generators (`np.random` or `random`).

---

## 3. What Was Wrong in the Previous Prediction?

### The Problem
When partial shipment dictionaries were submitted (for example, supplying only 8 attributes like `priority`, `distance_km`, `traffic_index`, `rainfall`, `visibility`, `maintenance_status`, `route_risk`, `documentation_complete`), the remaining 22 numeric features were initialized to **`0.0`**:

```python
# Flawed Fallback Behavior:
"capacity_kg": 0.0,
"average_rating": 0.0,
"warehouse_capacity": 0.0,
"current_utilization": 0.0,
"average_transit_days": 0.0,
"vehicle_age": 0.0
```

### How Decision Trees Were Tricked
In the trained Random Forest decision trees, decision nodes check thresholds such as:
- `capacity_kg <= 1000` ➔ Evaluates to True ➔ **Infers vehicle failure or zero payload capacity**.
- `average_rating <= 2.0` ➔ Evaluates to True ➔ **Infers unrated / blacklisted carrier**.
- `warehouse_capacity <= 500` ➔ Evaluates to True ➔ **Infers severe logistics warehouse breakdown**.

Because `capacity_kg = 0.0`, `average_rating = 0.0`, and `warehouse_capacity = 0.0` were passed into the model, the decision trees followed **extreme high-risk failure branches**. As a result, a low-risk, 250 km shipment was falsely predicted as **Delayed (51.66% delay probability)**.

---

## 4. How We Solved the Issue

### Solution: Dataset Median Imputation
Instead of filling missing numeric attributes with `0.0`, we analyzed the master training dataset (`df_master_with_null.xls`) and extracted the exact **dataset medians** for all numerical variables:

```python
DEFAULT_FEATURE_MEDIANS = {
    'weight_kg': 217.55,
    'volume_cbm': 0.82,
    'declared_value': 22526.13,
    'weight_per_unit': 5.79,
    'average_rating': 3.38,
    'fleet_size': 361.0,
    'years_of_service': 22.0,
    'capacity_kg': 20690.0,
    'vehicle_age': 13.0,
    'warehouse_capacity': 10868.0,
    'current_utilization': 69.0,
    'distance_km': 350.0,
    'average_transit_days': 16.0,
    'traffic_index': 53.0,
    'temperature': 15.6,
    'rainfall': 0.3,
    'humidity': 59.4,
    'wind_speed': 5.4,
    'visibility': 7.7,
    'dispatch_lead_time': 2.0,
    'Expected Transit Days': 16.0
}
```

### Corrected Result
With median imputation implemented in [src/preprocess.py](file:///c:/technetic_internship/src/preprocess.py), any unsupplied numeric attributes safely take realistic baseline values (`average_rating: 3.38`, `capacity_kg: 20690`, `warehouse_capacity: 10868`). 

For the user's scenario:
```python
scenario = {
    "priority": "Standard",
    "distance_km": 250,
    "traffic_index": 2,
    "rainfall": 5,
    "visibility": 12,
    "maintenance_status": "Good",
    "route_risk": "Low",
    "documentation_complete": True
}
```

The model now correctly evaluates normal logistics conditions and predicts:
```python
{
    "prediction": "On Time",
    "prediction_class": 0,
    "delay_probability": 24.97,
    "confidence": 75.03
}
```

---

## 5. How Prediction Uses the Dataset & Model Features

The 30 selected features passed into `random_forest_model.pkl` are generated from raw dataset fields as follows:

| Feature Name | Source in Dataset | Preprocessing / Feature Engineering Formula |
| :--- | :--- | :--- |
| `priority` | `priority` | `PRIORITY_MAP`: `'Standard' -> 1`, `'Express' -> 2`, `'Urgent' -> 3` |
| `weight_kg` | `weight_kg` | Direct numeric value (median fallback: `217.55`) |
| `volume_cbm` | `volume_cbm` | Direct numeric value (median fallback: `0.82`) |
| `declared_value` | `declared_value` | Direct numeric value (median fallback: `22526.13`) |
| `weight_per_unit` | `weight_per_unit` | Direct numeric value (median fallback: `5.79`) |
| `average_rating` | `average_rating` | Direct numeric value (median fallback: `3.38`) |
| `fleet_size` | `fleet_size` | Direct numeric value (median fallback: `361.0`) |
| `years_of_service` | `years_of_service` | Direct numeric value (median fallback: `22.0`) |
| `capacity_kg` | `capacity_kg` | Direct numeric value (median fallback: `20690.0`) |
| `maintenance_status` | `maintenance_status` | `MAINTENANCE_MAP`: `'Good' -> 3`, `'Due' -> 2`, `'Under Maintenance' -> 1` |
| `vehicle_age` | `vehicle_age` | Direct numeric value (median fallback: `13.0`) |
| `vehicle_utilization` | `weight_kg`, `capacity_kg` | Calculated Ratio: `weight_kg / capacity_kg` |
| `warehouse_capacity` | `warehouse_capacity` | Direct numeric value (median fallback: `10868.0`) |
| `current_utilization` | `current_utilization` | Direct numeric value (median fallback: `69.0`) |
| `distance_km` | `distance_km` | Direct numeric value (median fallback: `350.0`) |
| `average_transit_days` | `average_transit_days` | Direct numeric value (median fallback: `16.0`) |
| `route_risk` | `route_risk` | `ROUTE_RISK_MAP`: `'Low' -> 1`, `'Medium' -> 2`, `'High' -> 3` |
| `traffic_index` | `traffic_index` | Direct numeric value (median fallback: `53.0`) |
| `temperature` | `temperature` | Direct numeric value (median fallback: `15.6`) |
| `rainfall` | `rainfall` | Direct numeric value (median fallback: `0.3`) |
| `humidity` | `humidity` | Direct numeric value (median fallback: `59.4`) |
| `wind_speed` | `wind_speed` | Direct numeric value (median fallback: `5.4`) |
| `visibility` | `visibility` | Direct numeric value (median fallback: `7.7`) |
| `dispatch_lead_time` | `booking_date`, `ship_date` | Calculated Days: `(ship_date - booking_date).days` |
| `Expected Transit Days` | `ship_date`, `expected_delivery_date` | Calculated Days: `(expected_delivery_date - ship_date).days` |
| `value_density` | `declared_value`, `weight_kg` | Calculated Ratio: `declared_value / weight_kg` |
| `shipment_type_Import` | `shipment_type` | One-Hot Encoding: `1` if `shipment_type == 'Import'`, else `0` |
| `weather_condition_Fog` | `weather_condition` | One-Hot Encoding: `1` if `weather_condition == 'Fog'`, else `0` |
| `weather_condition_Storm` | `weather_condition` | One-Hot Encoding: `1` if `weather_condition == 'Storm'`, else `0` |
| `documentation_complete_True` | `documentation_complete` | Boolean Encoding: `1` if complete (`True`/`Yes`/`1`), else `0` |

---

## 6. Future Feature & Model Improvements

To further enhance prediction accuracy, system scalability, and operational usability, the following improvements are recommended:

1. **Pickled Pipeline Serialization (`pipeline.pkl`)**:
   - Wrap `SimpleImputer`, `StandardScaler`, and `RandomForestClassifier` into a unified `sklearn.pipeline.Pipeline` object to ensure feature transformations and imputations are serialized in a single artifact.
2. **Integration of Advanced Gradient Boosting Models**:
   - Train and benchmark **CatBoost** and **LightGBM**, which handle categorical features naturally and often achieve higher recall on imbalanced delay data.
3. **Real-Time Traffic & Weather API Integration**:
   - Integrate live weather (OpenWeatherMap API) and traffic congestion data (Google Maps Distance Matrix API) directly into `app/dashboard.py` to auto-populate weather and traffic features based on origin/destination coordinates.
4. **REST API Deployment (FastAPI)**:
   - Create a lightweight FastAPI service (`app/api.py`) allowing external enterprise ERP systems (SAP, Oracle Logistics) to query delay predictions via HTTP POST endpoints.
5. **Model Retraining Pipeline**:
   - Implement an automated retraining script (`src/train.py`) to periodically retrain the model on fresh shipment telemetry data and update `models/random_forest_model.pkl`.
