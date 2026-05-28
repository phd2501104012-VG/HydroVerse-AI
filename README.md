# 🌊 HydroVerse AI — Next-Gen Climate Hazard Intelligence Platform

**Research-grade platform for climate hazard detection, forecasting, and real-time monitoring across all 52 Madhya Pradesh districts.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io)
[![GEE](https://img.shields.io/badge/GEE-Google_Earth_Engine-yellow)](https://earthengine.google.com)

---

## Architecture

```
D:\cri/
├── config/            # Central configuration (settings, constants)
├── data/              # Data loaders (IMD, ERA5, CMIP6, SourceManager)
├── geospatial/        # Boundaries, pixel hazards, clustering, visualization
├── hazards/           # Flood, drought, heatwave, agri_stress, compound, detection
├── forecasting/       # ML (0-90d), CMIP6 (to 2040), blending, ensemble
├── models/            # Classical ML, DL (LSTM, ConvLSTM), TFT, ensemble, evaluator, explainer
├── validation/        # Historical validation, skill scores, reliability, confusion
├── realtime/          # GEE-based monitoring, alerts, anomaly detection
├── dashboard/         # Streamlit UI (dark glassmorphism theme)
│   ├── app.py         # Main application entry point
│   ├── config.py      # Dashboard-specific branding
│   ├── components/    # Sidebar, maps, charts, alerts, forecast, realtime panels
│   └── assets/        # CSS stylesheet
├── api/               # FastAPI REST + WebSocket server
├── exports/           # Report, data, figure export generators
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
└── setup.py           # Package configuration
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
python main.py dashboard

# 3. (Optional) Start REST API
python main.py api

# 4. (Optional) Run validation pipeline
python main.py validate
```

## Features

| Feature | Details |
|---------|---------|
| **Data Sources** | ERA5-Land (GEE), IMD Gridded NetCDF, auto-detect mode |
| **Hazard Detection** | WMO SPI-3 drought, IMD heatwave (≥40°C + 4.5°C departure), multi-day flood, VHI+NCDD agri-stress |
| **Compound Hazards** | 4×4 interaction matrix with multiplicative severity boost |
| **Pixel Intelligence** | Pixel-level severity, hotspot clustering (DBSCAN), extreme polygon extraction |
| **Forecasting** | ML (RF/XGBoost/LightGBM — 90d recursive), CMIP6 (8-GCM ensemble to 2040), clamped blend |
| **AI/ML Models** | Classical (RF, ET, GBM, XGB, LGBM), Deep (LSTM, ConvLSTM, CNN-LSTM), TFT Transformer |
| **Explainability** | SHAP, permutation importance, uncertainty quantification (MC dropout) |
| **Validation** | POD/FAR/HSS, Brier score, CRPS, reliability curves, confusion matrix (12 metrics) |
| **Real-time** | GEE (IMERG precip, MODIS NDVI/LST, SMAP soil moisture), cooldown-based alerts |
| **Dashboard** | Dark glassmorphism UI, 7 tabs, responsive gauges, interactive maps (folium) |
| **API** | RESTful FastAPI endpoints + WebSocket real-time push |
| **Scalability** | Dask chunking, parallel processing, 3-tier cache (memory/pickle/parquet) |

## Configuration

All system settings are centralized in `config/settings.py`:

```python
from config import CFG

CFG.active_data_source = DataSource.IMD  # Switch to IMD
CFG.forecast.horizon_days = 365           # Extend forecast horizon
CFG.gee.project_id = "my-project"         # GEE project
```

## CLI Usage

```
python main.py dashboard    # Launch Streamlit dashboard (port 8501)
python main.py api          # Launch FastAPI server (port 8000)
python main.py validate     # Run validation across districts
python main.py monitor      # Start real-time monitoring loop
```

## License

MIT
