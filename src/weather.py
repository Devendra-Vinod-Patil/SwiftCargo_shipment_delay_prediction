import os
import sys
import requests
import json

# Path resolution — src/weather.py lives inside src/, so project root is one level up
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_delay

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = "0bdc42c5a1ae434831c7310f04854edf"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def safe_float(val, default: float = 0.0) -> float:
    """Safely converts input value to float with default fallback for None or invalid values."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


def map_weather_condition(main_type: str, description: str) -> str:
    """
    Maps OpenWeatherMap main weather types and descriptions to the model's supported weather conditions:
    ['Clear', 'Cloudy', 'Rain', 'Fog', 'Storm', 'Snow']
    """
    main_type = str(main_type or "").lower()
    description = str(description or "").lower()

    if any(term in main_type or term in description for term in ["thunderstorm", "squall", "tornado", "gale", "storm", "hurricane"]):
        return "Storm"
    elif any(term in main_type or term in description for term in ["rain", "drizzle", "shower", "downpour"]):
        return "Rain"
    elif any(term in main_type or term in description for term in ["snow", "sleet", "blizzard", "ice", "freezing"]):
        return "Snow"
    elif any(term in main_type or term in description for term in ["fog", "mist", "smoke", "haze", "dust", "sand", "ash"]):
        return "Fog"
    elif any(term in main_type or term in description for term in ["cloud", "overcast"]):
        return "Cloudy"
    else:
        return "Clear"


def get_weather_forecast(city_name: str, api_key: str = OPENWEATHER_API_KEY) -> dict:
    """
    Fetches 5-day / 3-hour forecast telemetry from OpenWeatherMap API for hour-by-hour & day-by-day predictions.
    Returns structured 3-hour blocks with rain probabilities, rain volumes, and weather conditions.
    """
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(OPENWEATHER_FORECAST_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json() or {}
            forecast_list = data.get("list") or []
            
            parsed_entries = []
            for item in forecast_list:
                w_list = item.get("weather") or [{}]
                main_info = item.get("main") or {}
                wind_info = item.get("wind") or {}
                rain_obj = item.get("rain") or {}
                snow_obj = item.get("snow") or {}
                
                m_w = w_list[0].get("main", "Clear") if w_list else "Clear"
                w_d = w_list[0].get("description", "clear sky") if w_list else "clear sky"
                cond = map_weather_condition(m_w, w_d)
                
                r_3h = safe_float(rain_obj.get("3h"), 0.0)
                s_3h = safe_float(snow_obj.get("3h"), 0.0)
                precip_3h = max(r_3h, s_3h)
                pop = safe_float(item.get("pop"), 0.0) * 100.0  # percentage
                
                if precip_3h == 0.0 and cond in ["Rain", "Storm"]:
                    precip_3h = 3.5 if "light" in w_d.lower() else 5.0

                parsed_entries.append({
                    "datetime": item.get("dt_txt", ""),
                    "temp": round(safe_float(main_info.get("temp"), 25.0), 1),
                    "humidity": round(safe_float(main_info.get("humidity"), 50.0), 1),
                    "wind_speed": round(safe_float(wind_info.get("speed"), 3.0) * 3.6, 1),
                    "visibility": round(safe_float(item.get("visibility"), 10000.0) / 1000.0, 1),
                    "weather_condition": cond,
                    "description": w_d.title(),
                    "rain_3h": round(precip_3h, 2),
                    "rain_pop": round(pop, 1)
                })
            return {"success": True, "entries": parsed_entries, "error": None}
        return {"success": False, "entries": [], "error": f"Forecast API response ({response.status_code})"}
    except Exception as e:
        return {"success": False, "entries": [], "error": str(e)}


def get_live_weather_prediction(city_name: str, api_key: str = OPENWEATHER_API_KEY) -> dict:
    """
    Fetches real-time weather prediction data for a given city using OpenWeatherMap API
    and maps parameters to the exact feature inputs required by the shipment model.
    Includes automatic cross-verification with 3-hour forecast telemetry.
    """
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json() or {}
            main = data.get("main") or {}
            wind = data.get("wind") or {}
            weather_list = data.get("weather") or [{}]
            rain_info = data.get("rain") or {}
            snow_info = data.get("snow") or {}

            temp = safe_float(main.get("temp"), 25.0)
            humidity = safe_float(main.get("humidity"), 50.0)
            wind_speed = safe_float(wind.get("speed"), 3.0) * 3.6  # convert m/s to km/h
            visibility = safe_float(data.get("visibility"), 10000.0) / 1000.0  # convert m to km

            main_weather = weather_list[0].get("main", "Clear") if weather_list else "Clear"
            weather_desc = weather_list[0].get("description", "clear sky") if weather_list else "clear sky"
            weather_condition = map_weather_condition(main_weather, weather_desc)

            # Extraction of direct precipitation volume from API rain/snow objects
            rain_1h = safe_float(rain_info.get("1h"), 0.0)
            rain_3h = safe_float(rain_info.get("3h"), 0.0)
            snow_1h = safe_float(snow_info.get("1h"), 0.0)
            snow_3h = safe_float(snow_info.get("3h"), 0.0)
            measured_precip = max(rain_1h, rain_3h, snow_1h, snow_3h)

            # Query 24-hour Forecast Telemetry (sum of 8 x 3-hour forecast blocks over the 24-hour prediction window)
            fc_res = get_weather_forecast(city_name, api_key)
            rainfall_24h = 0.0
            max_pop_24h = 0.0
            fc_dominant_cond = weather_condition
            fc_24h_desc = ""

            if fc_res["success"] and fc_res["entries"]:
                fc_24h_entries = fc_res["entries"][:8]
                rainfall_24h = round(sum(e["rain_3h"] for e in fc_24h_entries), 2)
                max_pop_24h = max((e["rain_pop"] for e in fc_24h_entries), default=0.0)
                
                cond_counts = [e["weather_condition"] for e in fc_24h_entries]
                if "Storm" in cond_counts:
                    fc_dominant_cond = "Storm"
                elif "Rain" in cond_counts:
                    fc_dominant_cond = "Rain"
                elif "Snow" in cond_counts:
                    fc_dominant_cond = "Snow"
                elif "Fog" in cond_counts:
                    fc_dominant_cond = "Fog"
                elif "Cloudy" in cond_counts:
                    fc_dominant_cond = "Cloudy"

                first_desc = fc_24h_entries[0]["description"]
                fc_24h_desc = f"{first_desc} (24h Forecast Rain: {rainfall_24h}mm, {max_pop_24h:.0f}% max prob)"

            # Use 24-hour accumulated forecast rainfall prediction for model rainfall input
            if rainfall_24h > 0:
                rainfall = rainfall_24h
                weather_condition = fc_dominant_cond
                weather_desc = fc_24h_desc
            elif measured_precip > 0:
                rainfall = measured_precip
            elif weather_condition in ["Rain", "Storm"]:
                desc_lower = weather_desc.lower()
                if any(w in desc_lower for w in ["heavy", "extreme", "torrential", "thunderstorm"]):
                    rainfall = 25.0
                elif any(w in desc_lower for w in ["moderate"]):
                    rainfall = 10.0
                elif any(w in desc_lower for w in ["light", "drizzle", "shower"]):
                    rainfall = 3.5
                else:
                    rainfall = 5.0
            else:
                rainfall = 0.0

            return {
                "success": True,
                "city": data.get("name", city_name),
                "country": data.get("sys", {}).get("country", "") if isinstance(data.get("sys"), dict) else "",
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "wind_speed": round(wind_speed, 2),
                "visibility": round(visibility, 2),
                "rainfall": round(rainfall, 2),
                "rainfall_24h": round(rainfall_24h, 2),
                "weather_condition": weather_condition,
                "raw_description": weather_desc.title(),
                "error": None
            }

        else:
            # Handle API authentication / activation delay or non-200 responses with realistic fallback estimates
            err_msg = response.json().get("message", response.text) if response.headers.get("content-type") == "application/json" else response.text
            
            # Default realistic seasonal weather estimates for city querying
            return {
                "success": False,
                "city": city_name,
                "country": "N/A",
                "temperature": 25.0,
                "humidity": 60.0,
                "wind_speed": 12.0,
                "visibility": 8.0,
                "rainfall": 2.0,
                "weather_condition": "Clear",
                "raw_description": "API Fallback Estimate",
                "error": f"OpenWeatherMap API Response ({response.status_code}): {err_msg}"
            }

    except Exception as e:
        return {
            "success": False,
            "city": city_name,
            "country": "N/A",
            "temperature": 25.0,
            "humidity": 60.0,
            "wind_speed": 12.0,
            "visibility": 8.0,
            "rainfall": 2.0,
            "weather_condition": "Clear",
            "raw_description": "Network Fallback",
            "error": f"Connection error: {str(e)}"
        }


def predict_shipment_with_weather(city_name: str, shipment_data: dict = None) -> dict:
    """
    Combines OpenWeatherMap live weather prediction data with shipment attributes 
    to predict shipment delay.
    """
    weather_info = get_live_weather_prediction(city_name)

    if shipment_data is None:
        shipment_data = {
            "priority": "Standard",
            "distance_km": 350,
            "traffic_index": 25,
            "route_risk": "Low",
            "maintenance_status": "Good",
            "documentation_complete": True
        }

    # Inject live weather prediction values directly into shipment record
    merged_data = dict(shipment_data)
    merged_data["temperature"] = weather_info["temperature"]
    merged_data["humidity"] = weather_info["humidity"]
    merged_data["wind_speed"] = weather_info["wind_speed"]
    merged_data["visibility"] = weather_info["visibility"]
    merged_data["rainfall"] = weather_info["rainfall"]
    merged_data["weather_condition"] = weather_info["weather_condition"]

    # Execute Shipment Delay Prediction
    prediction_result = predict_delay(merged_data)

    return {
        "city": weather_info["city"],
        "weather": weather_info,
        "shipment_input": merged_data,
        "delay_prediction": prediction_result
    }


if __name__ == "__main__":
    print("=" * 75)
    print("  OPENWEATHERMAP API VERIFICATION & SHIPMENT DELAY PREDICTOR (src/weather.py)")
    print("=" * 75)
    print(f"API Key : {OPENWEATHER_API_KEY}")

    # Explicitly verify API Key Connectivity
    verify_url = f"{OPENWEATHER_BASE_URL}?q=London&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(verify_url, timeout=5)
        if resp.status_code == 200:
            print("API Key Status: VERIFIED & ACTIVE (200 OK)")
        else:
            print(f"API Key Status: FAILED ({resp.status_code}) - {resp.text}")
    except Exception as err:
        print(f"API Key Verification Error: {err}")

    test_cities = ["Mumbai", "London", "Cherrapunjee", "Hilo", "Seattle", "Tokyo"]

    for city in test_cities:
        res = predict_shipment_with_weather(city)
        w = res["weather"]
        p = res["delay_prediction"]

        print(f"\n[CITY] : {w['city']}, {w.get('country', '')}")
        print(f"  - Temperature      : {w['temperature']} °C")
        print(f"  - Humidity         : {w['humidity']} %")
        print(f"  - Wind Speed        : {w['wind_speed']} km/h")
        print(f"  - Visibility        : {w['visibility']} km")
        print(f"  - Rainfall          : {w['rainfall']} mm")
        print(f"  - Weather Condition: {w['weather_condition']} ({w['raw_description']})")
        print(f"  - Fetch Status     : {'SUCCESS' if w['success'] else 'FALLBACK'}")
        
        if w.get("error"):
            print(f"  - Notice           : {w['error']}")

        print(f"  -------------- SHIPMENT DELAY PREDICTION --------------")
        print(f"  - Prediction Status : {p['prediction']}")
        print(f"  - Delay Probability : {p['delay_probability']}%")
        print(f"  - Model Confidence  : {p['confidence']}%")

    print("\n" + "=" * 75)
