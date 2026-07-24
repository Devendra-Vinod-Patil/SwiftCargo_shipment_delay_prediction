import pandas as pd
import numpy as np

# Dataset medians derived from training data for fallback when partial inputs are provided
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

# Categorical Mappings
PRIORITY_MAP = {'Standard': 1, 'standard': 1, 1: 1, 'Express': 2, 'express': 2, 2: 2, 'Urgent': 3, 'urgent': 3, 3: 3}
MAINTENANCE_MAP = {'Good': 3, 'good': 3, 3: 3, 'Due': 2, 'due': 2, 2: 2, 'Under Maintenance': 1, 'under maintenance': 1, 1: 1}
ROUTE_RISK_MAP = {'Low': 1, 'low': 1, 1: 1, 'Medium': 2, 'medium': 2, 2: 2, 'High': 3, 'high': 3, 3: 3}
STATUS_MAP = {'Inactive': 0, 'inactive': 0, 0: 0, 'Active': 1, 'active': 1, 1: 1}


def preprocess_single_record(record: dict, selected_features: list) -> pd.DataFrame:
    """
    Preprocesses a shipment record dictionary into a DataFrame 
    aligned with the selected features required by the trained model.
    """
    if not isinstance(record, dict):
        record = {}

    rec = dict(record)

    # If input already provides all preprocessed selected features
    if selected_features and all(feat in rec for feat in selected_features):
        df_feat = pd.DataFrame([rec])[selected_features]
        return df_feat.astype(float)

    # 1. Handle Raw Dates
    booking_date = pd.to_datetime(rec.get('booking_date'), errors='coerce')
    ship_date = pd.to_datetime(rec.get('ship_date'), errors='coerce')
    expected_delivery_date = pd.to_datetime(rec.get('expected_delivery_date'), errors='coerce')

    if pd.notnull(ship_date) and pd.notnull(booking_date):
        dispatch_lead_time = max(0, (ship_date - booking_date).days)
    else:
        dispatch_lead_time = float(rec.get('dispatch_lead_time', DEFAULT_FEATURE_MEDIANS['dispatch_lead_time']))

    if pd.notnull(expected_delivery_date) and pd.notnull(ship_date):
        expected_transit_days = max(0, (expected_delivery_date - ship_date).days)
    elif rec.get('Expected Transit Days') is not None:
        expected_transit_days = float(rec['Expected Transit Days'])
    elif rec.get('expected_transit_days') is not None:
        expected_transit_days = float(rec['expected_transit_days'])
    else:
        # Dynamically estimate expected transit days based on distance if not explicitly provided
        dist_val = float(rec.get('distance_km', DEFAULT_FEATURE_MEDIANS['distance_km']))
        expected_transit_days = max(1.0, float(round(dist_val / 350.0, 1)))

    # 2. Numeric Attributes
    weight_kg = float(rec.get('weight_kg', DEFAULT_FEATURE_MEDIANS['weight_kg']))
    capacity_kg = float(rec.get('capacity_kg', DEFAULT_FEATURE_MEDIANS['capacity_kg']))
    declared_value = float(rec.get('declared_value', DEFAULT_FEATURE_MEDIANS['declared_value']))
    distance_km = float(rec.get('distance_km', DEFAULT_FEATURE_MEDIANS['distance_km']))

    vehicle_utilization = (weight_kg / capacity_kg) if capacity_kg > 0 else (DEFAULT_FEATURE_MEDIANS['weight_kg'] / DEFAULT_FEATURE_MEDIANS['capacity_kg'])
    value_density = (declared_value / weight_kg) if weight_kg > 0 else (DEFAULT_FEATURE_MEDIANS['declared_value'] / DEFAULT_FEATURE_MEDIANS['weight_kg'])

    # 3. Categorical Encodings
    priority = PRIORITY_MAP.get(rec.get('priority'), rec.get('priority', 1))
    maintenance_status = MAINTENANCE_MAP.get(rec.get('maintenance_status'), rec.get('maintenance_status', 3))
    route_risk = ROUTE_RISK_MAP.get(rec.get('route_risk'), rec.get('route_risk', 1))

    # 4. One-Hot & Boolean Flags
    shipment_type = str(rec.get('shipment_type', ''))
    shipment_type_Import = 1 if shipment_type.lower() == 'import' or rec.get('shipment_type_Import') == 1 else 0

    weather_condition = str(rec.get('weather_condition', ''))
    weather_condition_Fog = 1 if weather_condition.lower() == 'fog' or rec.get('weather_condition_Fog') == 1 else 0
    weather_condition_Storm = 1 if weather_condition.lower() == 'storm' or rec.get('weather_condition_Storm') == 1 else 0

    doc_complete_val = rec.get('documentation_complete', 1)
    if isinstance(doc_complete_val, str):
        documentation_complete_True = 1 if doc_complete_val.strip().lower() in ['yes', 'true', '1'] else 0
    else:
        documentation_complete_True = 1 if bool(doc_complete_val) or rec.get('documentation_complete_True') == 1 else 0

    # Build row dict
    row = {
        'priority': priority,
        'weight_kg': weight_kg,
        'volume_cbm': float(rec.get('volume_cbm', DEFAULT_FEATURE_MEDIANS['volume_cbm'])),
        'declared_value': declared_value,
        'weight_per_unit': float(rec.get('weight_per_unit', DEFAULT_FEATURE_MEDIANS['weight_per_unit'])),
        'average_rating': float(rec.get('average_rating', DEFAULT_FEATURE_MEDIANS['average_rating'])),
        'fleet_size': float(rec.get('fleet_size', DEFAULT_FEATURE_MEDIANS['fleet_size'])),
        'years_of_service': float(rec.get('years_of_service', DEFAULT_FEATURE_MEDIANS['years_of_service'])),
        'capacity_kg': capacity_kg,
        'maintenance_status': maintenance_status,
        'vehicle_age': float(rec.get('vehicle_age', DEFAULT_FEATURE_MEDIANS['vehicle_age'])),
        'vehicle_utilization': float(vehicle_utilization),
        'warehouse_capacity': float(rec.get('warehouse_capacity', DEFAULT_FEATURE_MEDIANS['warehouse_capacity'])),
        'current_utilization': float(rec.get('current_utilization', DEFAULT_FEATURE_MEDIANS['current_utilization'])),
        'distance_km': distance_km,
        'average_transit_days': float(rec.get('average_transit_days', DEFAULT_FEATURE_MEDIANS['average_transit_days'])),
        'route_risk': route_risk,
        'traffic_index': float(rec.get('traffic_index', DEFAULT_FEATURE_MEDIANS['traffic_index'])),
        'temperature': float(rec.get('temperature', DEFAULT_FEATURE_MEDIANS['temperature'])),
        'rainfall': float(rec.get('rainfall', DEFAULT_FEATURE_MEDIANS['rainfall'])),
        'humidity': float(rec.get('humidity', DEFAULT_FEATURE_MEDIANS['humidity'])),
        'wind_speed': float(rec.get('wind_speed', DEFAULT_FEATURE_MEDIANS['wind_speed'])),
        'visibility': float(rec.get('visibility', DEFAULT_FEATURE_MEDIANS['visibility'])),
        'dispatch_lead_time': float(dispatch_lead_time),
        'Expected Transit Days': float(expected_transit_days),
        'value_density': float(value_density),
        'shipment_type_Import': shipment_type_Import,
        'weather_condition_Fog': weather_condition_Fog,
        'weather_condition_Storm': weather_condition_Storm,
        'documentation_complete_True': documentation_complete_True
    }

    df = pd.DataFrame([row])
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0.0

    return df[selected_features]


def preprocess_batch(df_batch: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Preprocesses a DataFrame batch of shipment records into feature DataFrame.
    """
    records = df_batch.to_dict(orient='records')
    dfs = [preprocess_single_record(r, selected_features) for r in records]
    return pd.concat(dfs, ignore_index=True)
