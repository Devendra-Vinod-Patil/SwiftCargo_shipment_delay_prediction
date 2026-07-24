# Shipment Delay Prediction Process & Analysis

This document provides a comprehensive overview of the **Shipment Delay Prediction Pipeline**, explaining the feature engineering, categorical encodings, data preprocessing, model inference, and the resolution of the previous fallback prediction issue.

---

## 1. What Was Wrong in the Previous Prediction?

### The Root Cause
When evaluating partial input records (for example, providing only 8 fields out of 30 required model features), omitted numerical attributes were initially filled with `0.0`.

```python
# Previous Fallback (Flawed):
"capacity_kg": 0.0,
"average_rating": 0.0,
"warehouse_capacity": 0.0,
"current_utilization": 0.0,
"average_transit_days": 0.0,
"vehicle_age": 0.0
```

### Impact on Random Forest Decision Trees
In a trained **Random Forest Classifier**, decision tree split nodes evaluate thresholds such as:
- `capacity_kg <= 1000` (Infers zero vehicle capacity or severe overload)
- `average_rating <= 2.0` (Infers unrated or poor-performing carrier)
- `warehouse_capacity <= 500` (Infers missing/inadequate logistics storage)

Setting missing numeric features to `0.0` caused decision trees to follow **high-risk anomaly paths**, falsely predicting a **Delayed** status (51.66% delay probability) for a low-risk, short-distance (250 km) shipment with clear weather and good vehicle maintenance.

### The Resolution
By analyzing the training dataset (`df_master_with_null.xls`), we extracted the exact **dataset medians** for all numerical columns:

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

When partial input records are passed, missing numeric attributes default to these realistic baseline medians. As a result, the low-risk scenario accurately predicts **On Time** with a **24.97% delay probability** (**75.03% confidence**).

---

## 2. The Complete Prediction Process Pipeline

The prediction workflow follows a systematic 7-step pipeline:

```
[Raw User Input Dict] 
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Input Validation & Format Check                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Date Parsing & Lead Time Computation                │
│   - dispatch_lead_time = ship_date - booking_date           │
│   - Expected Transit Days = expected_delivery - ship_date   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Derived Ratio Calculations                          │
│   - vehicle_utilization = weight_kg / capacity_kg           │
│   - value_density = declared_value / weight_kg              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Categorical Mapping (Ordinal Encodings)             │
│   - priority: Standard (1), Express (2), Urgent (3)         │
│   - maintenance_status: Good (3), Due (2), Maintenance (1)  │
│   - route_risk: Low (1), Medium (2), High (3)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: One-Hot & Boolean Encodings                         │
│   - shipment_type_Import, weather_condition_Fog, etc.       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Dataset Median Imputation for Unsupplied Features    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Random Forest Model Inference & Output Formatting   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Feature Definitions & Default Fallbacks

| Feature Name | Type | Description | Default Fallback Value |
| :--- | :--- | :--- | :--- |
| `priority` | Categorical (Ordinal) | Order urgency (`Standard`: 1, `Express`: 2, `Urgent`: 3) | `1` (`Standard`) |
| `weight_kg` | Numeric | Total shipment weight in kg | `217.55` |
| `volume_cbm` | Numeric | Total shipment volume in cubic meters | `0.82` |
| `declared_value` | Numeric | Declared monetary value of cargo | `22526.13` |
| `weight_per_unit` | Numeric | Average weight per item unit | `5.79` |
| `average_rating` | Numeric | Carrier performance rating (1.0 to 5.0) | `3.38` |
| `fleet_size` | Numeric | Total vehicles in carrier fleet | `361.0` |
| `years_of_service` | Numeric | Carrier operational experience in years | `22.0` |
| `capacity_kg` | Numeric | Maximum vehicle cargo weight capacity | `20690.0` |
| `maintenance_status` | Categorical (Ordinal) | Vehicle condition (`Good`: 3, `Due`: 2, `Maintenance`: 1) | `3` (`Good`) |
| `vehicle_age` | Numeric | Age of the transit vehicle in years | `13.0` |
| `vehicle_utilization` | Numeric (Ratio) | `weight_kg / capacity_kg` | `weight / capacity` |
| `warehouse_capacity` | Numeric | Storage capacity of dispatch warehouse | `10868.0` |
| `current_utilization` | Numeric | Percentage warehouse utilization (0 to 100) | `69.0` |
| `distance_km` | Numeric | Total route distance in kilometers | `350.0` |
| `average_transit_days` | Numeric | Historical average transit days for route | `16.0` |
| `route_risk` | Categorical (Ordinal) | Route risk assessment (`Low`: 1, `Medium`: 2, `High`: 3) | `1` (`Low`) |
| `traffic_index` | Numeric | Real-time congestion index (0 to 100) | `53.0` |
| `temperature` | Numeric | Ambient temperature in Celsius | `15.6` |
| `rainfall` | Numeric | Rainfall amount in mm | `0.3` |
| `humidity` | Numeric | Relative humidity percentage | `59.4` |
| `wind_speed` | Numeric | Wind speed in km/h | `5.4` |
| `visibility` | Numeric | Visibility distance in km | `7.7` |
| `dispatch_lead_time` | Numeric (Days) | Days between booking date and ship date | `2.0` |
| `Expected Transit Days` | Numeric (Days) | Days between ship date and expected delivery date | `16.0` |
| `value_density` | Numeric (Ratio) | `declared_value / weight_kg` | `value / weight` |
| `shipment_type_Import` | Binary | `1` if `shipment_type` is `'Import'`, else `0` | `0` |
| `weather_condition_Fog` | Binary | `1` if `weather_condition` is `'Fog'`, else `0` | `0` |
| `weather_condition_Storm` | Binary | `1` if `weather_condition` is `'Storm'`, else `0` | `0` |
| `documentation_complete_True` | Binary | `1` if documentation complete (`True`/`Yes`/`1`), else `0` | `1` |

---

## 4. How to Use the Prediction Function

### Basic Python Usage

```python
from predict import predict_delay

# Define a shipment record
shipment_record = {
    "priority": "Standard",
    "distance_km": 250,
    "traffic_index": 2,
    "rainfall": 5,
    "visibility": 12,
    "maintenance_status": "Good",
    "route_risk": "Low",
    "documentation_complete": True
}

# Run prediction
result = predict_delay(shipment_record)

print(result)
```

### Sample Output Format

```python
{
    "prediction": "On Time",
    "prediction_class": 0,
    "delay_probability": 24.97,
    "confidence": 75.03
}
```

- **`prediction`**: Human-readable label (`"On Time"` or `"Delayed"`).
- **`prediction_class`**: Binary classification (`0` for On Time, `1` for Delayed).
- **`delay_probability`**: Probability of shipment delay expressed as a percentage (`0.0%` to `100.0%`).
- **`confidence`**: Statistical confidence percentage of the predicted class (`50.0%` to `100.0%`).

---

## 5. Verification & Testing

The prediction pipeline is verified using automated unit tests in [test_predict.py](file:///c:/technetic_internship/test_predict.py):

```powershell
python test_predict.py
```

All 13 test scenarios and edge-case validations pass with 100% accuracy.
