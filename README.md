# 🚚 Shipment Delay Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Machine Learning solution for predicting shipment delays in logistics and supply chain management.**

Real-time weather integration via **OpenWeatherMap API** · **Random Forest Classifier** · **71.28% Accuracy**

[🚀 Live Demo](#-deploy-on-streamlit-cloud) · [📖 Documentation](#-documentation) · [⚡ Quick Start](#-quick-start)

</div>

---

## ✨ Features

- 🌤️ **Live Weather Integration** — Fetches real-time weather for any city via OpenWeatherMap API
- 🤖 **ML Prediction Engine** — Random Forest model with RFECV feature selection (30 key features)
- 📊 **Interactive Dashboard** — Streamlit web UI with charts, batch predictions, and forecasts
- 📦 **Batch Processing** — Upload CSV files for bulk shipment delay predictions
- 🔮 **24-Hour Weather Forecast** — Route weather outlook to anticipate future delays

---

## 📁 Project Structure

```text
shipment-delay-predictor/
├── .streamlit/
│   └── config.toml               ← Streamlit theme & server config
├── app/
│   └── dashboard.py              ← Main Streamlit dashboard (UI)
├── src/
│   ├── __init__.py
│   ├── predict.py                ← Prediction engine & batch processor
│   └── preprocess.py             ← Feature engineering & data preprocessing
├── models/
│   ├── random_forest_model.pkl   ← Trained Random Forest (71.28% accuracy)
│   ├── scaler.pkl                ← StandardScaler artifact
│   └── selected_features.pkl    ← 30 RFECV-selected features
├── data/
│   └── sample_batch.csv          ← Sample CSV for batch predictions
├── notebooks/
│   ├── data_overview.ipynb       ← EDA & data exploration
│   ├── data_with_null_value_treatment.ipynb ← Null handling strategy
│   ├── model_building.ipynb      ← Model training & evaluation
│   └── preprocessing.ipynb      ← Feature engineering pipeline
├── docs/
│   ├── project.md                ← Technical project documentation
│   ├── prediction.md             ← Prediction engine deep-dive
│   ├── imputer.md                ← Imputation strategy analysis
│   ├── report.md                 ← Full project report
│   └── validation_report.md     ← System validation audit
├── tests/
│   └── test_predict.py           ← 13 automated unit test scenarios
├── src/weather.py                  ← OpenWeatherMap live weather predictor
├── streamlit_app.py              ← Root entry point (Streamlit Cloud)
├── requirements.txt              ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/shipment-delay-predictor.git
cd shipment-delay-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

> Or alternatively: `streamlit run app/dashboard.py`

### 4. Run the Live Weather Predictor (CLI)

```bash
python src/weather.py
```

### 5. Run Automated Tests

```bash
python -m pytest tests/test_predict.py -v
```

---

## 🌤️ OpenWeatherMap Integration

Using the OpenWeatherMap API, the system fetches **live weather** for any destination city and extracts the 6 weather parameters required by the model:

| Parameter | Description |
|-----------|-------------|
| `temperature` | Live temperature (°C) |
| `humidity` | Humidity (%) |
| `wind_speed` | Wind speed (km/h) |
| `visibility` | Visibility (km) |
| `rainfall` | Rainfall (mm) |
| `weather_condition` | Mapped to: Clear / Cloudy / Rain / Fog / Storm / Snow |

**Python API Example:**

```python
from src.weather import predict_shipment_with_weather

result = predict_shipment_with_weather("Mumbai")
print("Weather:", result["weather"])
print("Delay Prediction:", result["delay_prediction"])
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Model** | Random Forest Classifier (RFECV + Tuned) |
| **Accuracy** | **71.28%** |
| **Precision** | **73.68%** |
| **ROC-AUC** | **72.46%** |
| **Features Selected** | 30 Key Predictors |
| **Training Strategy** | RFECV + GridSearchCV |

---

## 🚀 Deploy on Streamlit Cloud

1. **Fork** this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New App** → select your forked repo
4. Set **Main file path** to: `streamlit_app.py`
5. Click **Deploy** — your app will be live in minutes!

> [!NOTE]
> No extra configuration needed. The `.streamlit/config.toml` handles all theme settings automatically.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [docs/project.md](docs/project.md) | Full technical project walkthrough |
| [docs/prediction.md](docs/prediction.md) | Prediction engine internals |
| [docs/imputer.md](docs/imputer.md) | Null value handling strategy |
| [docs/report.md](docs/report.md) | Comprehensive project report |
| [docs/validation_report.md](docs/validation_report.md) | System validation audit |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core language |
| **Streamlit** | Interactive web dashboard |
| **scikit-learn 1.6.1** | ML model (Random Forest + RFECV) |
| **Pandas / NumPy** | Data processing |
| **Altair** | Interactive charts |
| **OpenWeatherMap API** | Live weather data |
| **joblib** | Model serialization |


<div align="center">

Built as part of **Technetic Internship** · Made with ❤️ using Python & Streamlit

</div>
