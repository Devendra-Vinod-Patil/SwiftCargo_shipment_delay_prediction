import os
import sys
import unittest

# Ensure project root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_delay, selected_features
from src.preprocess import preprocess_single_record


class TestPredictDelay(unittest.TestCase):

    def setUp(self):
        self.sample_raw_record = {
            "booking_date": "2024-01-01",
            "ship_date": "2024-01-02",
            "expected_delivery_date": "2024-01-08",
            "priority": "Standard",
            "weight_kg": 120,
            "volume_cbm": 2.5,
            "declared_value": 50000,
            "insurance": 1,
            "fragile": 0,
            "hazardous": 0,
            "perishable": 0,
            "temperature_controlled": 0,
            "fragile_product": 0,
            "customer_status": "Active",
            "weight_per_unit": 5,
            "average_rating": 4.5,
            "fleet_size": 100,
            "years_of_service": 8,
            "capacity_kg": 500,
            "maintenance_status": "Good",
            "vehicle_age": 3,
            "warehouse_capacity": 5000,
            "current_utilization": 70,
            "distance_km": 350,
            "average_transit_days": 4,
            "route_risk": "Low",
            "traffic_index": 2,
            "temperature": 30,
            "rainfall": 10,
            "humidity": 60,
            "wind_speed": 12,
            "visibility": 8,
            "customs_required": 0,
            "documentation_complete": "Yes",
            "inspection_required": "No",
            "warehouse_type": "Regional",
            "shipping_mode": "Road",
            "shipment_type": "Domestic",
            "customer_type": "Business",
            "industry": "Retail",
            "category": "Electronics",
            "carrier_type": "Private",
            "vehicle_type": "Truck",
            "fuel_type": "Diesel",
            "weather_condition": "Clear",
            "cargo_type": "General"
        }

    # --- 10 Detailed Test Scenarios ---

    def test_scenario_1_normal_shipment(self):
        scenario = self.sample_raw_record.copy()
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_2_best_traffic_conditions(self):
        scenario = self.sample_raw_record.copy()
        scenario["traffic_index"] = 50
        scenario["visibility"] = 10
        scenario["rainfall"] = 0
        scenario["humidity"] = 40
        scenario["wind_speed"] = 2
        scenario["temperature"] = 25
        scenario["distance_km"] = 100
        scenario["current_utilization"] = 35
        scenario["route_risk"] = "Low"
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_3_heavy_traffic(self):
        scenario = self.sample_raw_record.copy()
        scenario["traffic_index"] = 100
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_4_heavy_rain(self):
        scenario = self.sample_raw_record.copy()
        scenario["rainfall"] = 300
        scenario["humidity"] = 95
        scenario["visibility"] = 2
        scenario["wind_speed"] = 70
        scenario["temperature"] = 20
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_5_long_distance(self):
        scenario = self.sample_raw_record.copy()
        scenario["distance_km"] = 3000
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_6_heavy_shipment(self):
        scenario = self.sample_raw_record.copy()
        scenario["weight_kg"] = 450
        scenario["declared_value"] = 200000
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_7_high_warehouse_utilization(self):
        scenario = self.sample_raw_record.copy()
        scenario["current_utilization"] = 98
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_8_high_route_risk(self):
        scenario = self.sample_raw_record.copy()
        scenario["route_risk"] = "High"
        scenario["traffic_index"] = 90
        scenario["distance_km"] = 2500
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_9_import_shipment_storm(self):
        scenario = self.sample_raw_record.copy()
        scenario["shipment_type"] = "Import"
        scenario["weather_condition"] = "Storm"
        scenario["traffic_index"] = 95
        scenario["rainfall"] = 250
        scenario["visibility"] = 1
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    def test_scenario_10_worst_case(self):
        scenario = self.sample_raw_record.copy()
        scenario["traffic_index"] = 100
        scenario["visibility"] = 1
        scenario["rainfall"] = 300
        scenario["humidity"] = 95
        scenario["wind_speed"] = 70
        scenario["temperature"] = 45
        scenario["distance_km"] = 3000
        scenario["current_utilization"] = 98
        scenario["weight_kg"] = 450
        scenario["declared_value"] = 200000
        scenario["route_risk"] = "High"
        scenario["shipment_type"] = "Import"
        scenario["weather_condition"] = "Storm"
        result = predict_delay(scenario)
        self.assertIn(result["prediction"], ["Delayed", "On Time"])

    # --- Edge Cases & Validation Tests ---

    def test_none_input(self):
        result = predict_delay(None)
        self.assertEqual(result["prediction"], "Error")

    def test_empty_dict_input(self):
        result = predict_delay({})
        self.assertEqual(result["prediction"], "Error")

    def test_feature_alignment(self):
        df_proc = preprocess_single_record(self.sample_raw_record, selected_features)
        self.assertEqual(list(df_proc.columns), selected_features)
        self.assertEqual(len(df_proc.columns), 30)


def run_all_scenarios():
    tester = TestPredictDelay()
    tester.setUp()
    sample_record = tester.sample_raw_record

    print("\n" + "=" * 65)
    print("      SHIPMENT DELAY PREDICTION - 10 TEST SCENARIOS")
    print("=" * 65)

    # Scenario 1 : Normal Shipment
    scenario = sample_record.copy()
    result = predict_delay(scenario)
    print("\nScenario 1 - Normal Shipment")
    print(result)

    # Scenario 2 : Low Traffic (Best)
    scenario = sample_record.copy()
    scenario["traffic_index"] = 50
    scenario["visibility"] = 10
    scenario["rainfall"] = 0
    scenario["humidity"] = 40
    scenario["wind_speed"] = 2
    scenario["temperature"] = 25
    scenario["distance_km"] = 100
    scenario["current_utilization"] = 35
    scenario["route_risk"] = "Low"
    result = predict_delay(scenario)
    print("\nScenario 2 - Best Traffic Conditions")
    print(result)

    # Scenario 3 : Heavy Traffic
    scenario = sample_record.copy()
    scenario["traffic_index"] = 100
    result = predict_delay(scenario)
    print("\nScenario 3 - Heavy Traffic")
    print(result)

    # Scenario 4 : Heavy Rain
    scenario = sample_record.copy()
    scenario["rainfall"] = 300
    scenario["humidity"] = 95
    scenario["visibility"] = 2
    scenario["wind_speed"] = 70
    scenario["temperature"] = 20
    result = predict_delay(scenario)
    print("\nScenario 4 - Heavy Rain")
    print(result)

    # Scenario 5 : Long Distance
    scenario = sample_record.copy()
    scenario["distance_km"] = 3000
    result = predict_delay(scenario)
    print("\nScenario 5 - Long Distance")
    print(result)

    # Scenario 6 : Heavy Shipment
    scenario = sample_record.copy()
    scenario["weight_kg"] = 450
    scenario["declared_value"] = 200000
    result = predict_delay(scenario)
    print("\nScenario 6 - Heavy Shipment")
    print(result)

    # Scenario 7 : High Warehouse Utilization
    scenario = sample_record.copy()
    scenario["current_utilization"] = 98
    result = predict_delay(scenario)
    print("\nScenario 7 - High Warehouse Utilization")
    print(result)

    # Scenario 8 : High Route Risk
    scenario = sample_record.copy()
    scenario["route_risk"] = "High"
    scenario["traffic_index"] = 90
    scenario["distance_km"] = 2500
    result = predict_delay(scenario)
    print("\nScenario 8 - High Route Risk")
    print(result)

    # Scenario 9 : Import Shipment + Bad Weather
    scenario = sample_record.copy()
    scenario["shipment_type"] = "Import"
    scenario["weather_condition"] = "Storm"
    scenario["traffic_index"] = 95
    scenario["rainfall"] = 250
    scenario["visibility"] = 1
    result = predict_delay(scenario)
    print("\nScenario 9 - Import Shipment During Storm")
    print(result)

    # Scenario 10 : Worst Case
    scenario = sample_record.copy()
    scenario["traffic_index"] = 100
    scenario["visibility"] = 1
    scenario["rainfall"] = 300
    scenario["humidity"] = 95
    scenario["wind_speed"] = 70
    scenario["temperature"] = 45
    scenario["distance_km"] = 3000
    scenario["current_utilization"] = 98
    scenario["weight_kg"] = 450
    scenario["declared_value"] = 200000
    scenario["route_risk"] = "High"
    scenario["shipment_type"] = "Import"
    scenario["weather_condition"] = "Storm"
    result = predict_delay(scenario)
    print("\nScenario 10 - Worst Case")
    print(result)

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_all_scenarios()
    print("\nRunning Automated Unit Tests:")
    unittest.main()
