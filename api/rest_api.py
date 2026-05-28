"""FastAPI REST API for programmatic access to hazard forecasts and alerts."""

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _has_fastapi = True
except ImportError:
    _has_fastapi = False

from typing import Optional, List, Dict, Any
from datetime import datetime, date
import pandas as pd
import json

from config import CFG, DataSource
from geospatial.boundaries import DistrictBoundaries
from hazards.detection import HazardDetector
from hazards.categories import HazardClassifier
from data.source_manager import DataSourceManager
from forecasting.daily_forecast import DailyForecastEngine
from validation.historical_validation import HistoricalValidator
from realtime.alerts import AlertEngine
from utils import get_logger

logger = get_logger(__name__)

_APP: Optional[Any] = None


def get_app() -> Optional[Any]:
    global _APP
    if not _has_fastapi:
        logger.warning("FastAPI not installed. REST API unavailable.")
        return None
    if _APP is None:
        _APP = FastAPI(
            title="HydroVerse AI API",
            description="Climate Hazard Intelligence Platform REST API",
            version="2.0.0",
        )
        _APP.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        _register_routes(_APP)
    return _APP


def _init_resources():
    boundaries = DistrictBoundaries()
    ds_manager = DataSourceManager()
    detector = HazardDetector()
    classifier = HazardClassifier()
    forecast_engine = DailyForecastEngine()
    alert_engine = AlertEngine()
    return boundaries, ds_manager, detector, classifier, forecast_engine, alert_engine


def _register_routes(app) -> None:
    boundaries, ds_manager, detector, classifier, forecast_engine, alert_engine = _init_resources()

    @app.get("/")
    def root():
        return {"service": "HydroVerse AI API", "version": "2.0.0", "status": "operational"}

    @app.get("/districts")
    def list_districts():
        return {"districts": boundaries.district_names}

    @app.get("/districts/{district}/hazards")
    def get_hazards(
        district: str,
        start: str = Query("2000-01-01"),
        end: str = Query("2025-12-31"),
        source: str = Query("auto"),
    ):
        if source == "imd":
            CFG.active_data_source = DataSource.IMD
        elif source == "era5":
            CFG.active_data_source = DataSource.ERA5
        else:
            CFG.active_data_source = DataSource.AUTO

        try:
            data = ds_manager.get_district_timeseries(district, ["tmax", "tmin", "precip"])
            lcc = detector.compute_climate_indices(data)
            hazards = detector.detect_all(lcc)
            return hazards.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/districts/{district}/forecast")
    def get_forecast(
        district: str,
        target: str = Query("tmax"),
        horizon_days: int = Query(365, ge=1, le=5475),
    ):
        try:
            forecasts = forecast_engine.generate_forecast(district, horizon_days=horizon_days)
            fc = forecasts.get((district, target))
            if fc is None or fc.empty:
                raise HTTPException(status_code=404, detail=f"No forecast for {district}/{target}")
            return fc.to_dict(orient="records")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/alerts")
    def get_alerts(severity_min: float = Query(0, ge=0, le=100)):
        try:
            data = ds_manager.get_multi_district_data(boundaries.district_names[:5], ["tmax", "tmin", "precip"])
            lcc = detector.compute_climate_indices(data)
            hazards = detector.detect_all(lcc)
            alerts = alert_engine.evaluate_alerts(hazards)
            alerts_df = alert_engine.alerts_to_dataframe(alerts)
            if severity_min > 0:
                alerts_df = alerts_df[alerts_df["severity"] >= severity_min]
            return alerts_df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/validation/{district}")
    def get_validation(district: str):
        try:
            return {"district": district, "message": "Validation endpoint - run validation pipeline separately"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check():
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "districts_loaded": len(boundaries.district_names),
            "data_source": CFG.active_data_source.value,
        }


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    if not _has_fastapi:
        logger.error("Cannot start API: FastAPI not installed")
        print("Install dependencies: pip install fastapi uvicorn")
        return

    import uvicorn
    logger.info(f"Starting HydroVerse AI API server on {host}:{port}")
    uvicorn.run(get_app(), host=host, port=port)
