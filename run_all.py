"""Run the complete HydroVerse AI pipeline end-to-end.
Mirrors the original notebook: ERA5 daily + IMD + MODIS/SMAP + CMIP6 — all combined.

Usage:
    python run_all.py                         # 5 districts (like original notebook)
    python run_all.py --all-districts         # All 51 MP districts (slow)
    python run_all.py --districts Bhopal,Indore
    python run_all.py --skip-ml
    python run_all.py --skip-cmip6
    python run_all.py --dashboard
"""

import sys, os, json, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config import CFG
from utils import get_logger
from utils.gee_utils import init_gee

logger = get_logger("run_all")


def parse_args():
    p = argparse.ArgumentParser(description="HydroVerse AI full pipeline")
    p.add_argument("--all-districts", action="store_true")
    p.add_argument("--districts", type=str)
    p.add_argument("--skip-ml", action="store_true")
    p.add_argument("--skip-cmip6", action="store_true")
    p.add_argument("--skip-satellite", action="store_true")
    p.add_argument("--dashboard", action="store_true")
    return p.parse_args()


def _fetch_era5_district(era5_loader, district: str, var: str) -> pd.Series:
    """Fetch ERA5 daily data for one variable, return Series indexed by date."""
    try:
        df = era5_loader.fetch_variable(var)
        if df.empty:
            return pd.Series(dtype=float)
        sub = df[df["district"] == district]
        if sub.empty:
            return pd.Series(dtype=float)
        s = sub.set_index("date")["value"]
        s.index = pd.to_datetime(s.index)
        s.name = f"{var}_era5"
        return s
    except Exception as e:
        logger.warning(f"    ERA5 {var} failed: {e}")
        return pd.Series(dtype=float)


def _fetch_imd_district(bounds_obj, imd_loader, district: str, var: str) -> pd.Series:
    """Load IMD NetCDF data for one district."""
    try:
        gdf = bounds_obj.load()
        if gdf.empty:
            return pd.Series(dtype=float)
        ts = imd_loader.district_timeseries(var, gdf, district)
        if ts is None:
            return pd.Series(dtype=float)
        ts.name = f"{var}_imd"
        return ts
    except Exception as e:
        logger.warning(f"    IMD {var} failed: {e}")
        return pd.Series(dtype=float)


def _merge_district_data(era5: dict, imd: dict, sat: pd.DataFrame, district: str) -> pd.DataFrame:
    """Merge ERA5 + IMD + satellite into single daily DataFrame."""
    parts = []
    for s in era5.values():
        if not s.empty:
            parts.append(s)
    for s in imd.values():
        if not s.empty:
            parts.append(s)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, axis=1)
    merged.index.name = "date"

    # Merge satellite data
    if not sat.empty:
        sat_sub = sat[sat["district"] == district]
        if not sat_sub.empty:
            sat_pivot = sat_sub.pivot_table(
                index="date", columns="variable", values="value", aggfunc="mean"
            )
            sat_pivot.index = pd.to_datetime(sat_pivot.index)
            merged = merged.join(sat_pivot, how="left")

    return merged.reset_index()


def run_pipeline(args):
    t_start = datetime.now()
    logger.info("=" * 60)
    logger.info("HydroVerse AI — Full Pipeline (ERA5 + IMD + Satellite + CMIP6)")
    logger.info(f"Started: {t_start.isoformat()}")
    logger.info("=" * 60)

    # --- Step 1: Init GEE ---
    logger.info("[1/8] Initializing Google Earth Engine...")
    try:
        init_gee_result = init_gee(project=CFG.gee.project)
    except Exception:
        init_gee_result = False
        logger.warning("GEE init failed (some sources unavailable)")

    # --- Step 2: Load boundaries ---
    logger.info("[2/8] Loading district boundaries...")
    from geospatial.boundaries import DistrictBoundaries
    bounds = DistrictBoundaries()
    if args.all_districts:
        districts = bounds.district_names
    elif args.districts:
        districts = [d.strip() for d in args.districts.split(",")]
    else:
        districts = bounds.district_names[:5]
    logger.info(f"  {len(districts)} districts: {', '.join(districts[:5])}{'...' if len(districts) > 5 else ''}")

    # --- Step 3: Create GEE FeatureCollection ---
    from data.era5_loader import ERA5Loader
    from data.cmip6_loader import CMIP6Loader
    ee_fc = bounds.get_ee_fc(districts)
    era5_loader = ERA5Loader(ee_fc) if ee_fc else None
    cmip6_loader = CMIP6Loader(ee_fc) if ee_fc else None

    # --- Step 4: Load IMD ---
    from data.imd_loader import IMDLoader
    imd_loader = IMDLoader() if Path(CFG.imd.precip_dir).exists() else None
    if imd_loader:
        logger.info("  IMD NetCDF loader ready (local)")
    else:
        logger.info("  IMD directory not found — ERA5 only")

    # --- Step 5: Satellite data (historical fetch via GEE) ---
    sat_data = pd.DataFrame()
    if not args.skip_satellite and ee_fc is not None and init_gee_result:
        logger.info("[3/8] Fetching historical satellite data (MODIS NDVI, SMAP)...")
        from config import constants
        from utils.gee_utils import chunked_fetch

        try:
            # MODIS NDVI (2000-present, 16-day composites → use large chunks)
            ndvi_cfg = constants.REALTIME_DATASETS["ndvi"]
            today_str = datetime.now().strftime("%Y-%m-%d")
            ndvi_df = chunked_fetch(ee_fc, ndvi_cfg["collection"], [ndvi_cfg["band"]],
                                    "2000-01-01", today_str, chunk_days=1500)
            if not ndvi_df.empty:
                ndvi_melt = ndvi_df.melt(id_vars=["district", "date"], value_name="value")
                ndvi_melt["value"] = ndvi_melt["value"] * ndvi_cfg["scale"]
                ndvi_melt["variable"] = "ndvi"
                sat_data = pd.concat([sat_data, ndvi_melt[["district", "date", "variable", "value"]]])
                logger.info(f"  NDVI: {len(ndvi_melt)} records")

            # ERA5-Land soil moisture (daily, 1 image/day)
            sm_cfg = constants.REALTIME_DATASETS["soil_moisture"]
            today_str = datetime.now().strftime("%Y-%m-%d")
            sm_df = chunked_fetch(ee_fc, sm_cfg["collection"], [sm_cfg["band"]],
                                  "2000-01-01", today_str)
            if not sm_df.empty:
                sm_melt = sm_df.melt(id_vars=["district", "date"], value_name="value")
                sm_melt["value"] = sm_melt["value"] * sm_cfg["scale"]
                sm_melt["variable"] = "soil_moisture"
                sat_data = pd.concat([sat_data, sm_melt[["district", "date", "variable", "value"]]])
                logger.info(f"  Soil moisture: {len(sm_melt)} records")

            if sat_data.empty:
                logger.info("  No satellite data available")
            else:
                logger.info(f"  Total satellite records: {len(sat_data)}")
        except Exception as e:
            logger.warning(f"  Satellite fetch failed: {e}")

    # --- Step 6: Process each district (ERA5 + IMD combined) ---
    logger.info("[4/8] Processing districts — fetching ERA5 daily + IMD...")
    ALL_ERA5_VARS = ["tmax", "tmin", "precip"]
    ALL_IMD_VARS = ["tmax", "tmin", "precip"]

    from hazards.detection import HazardDetector
    from hazards.compound import CompoundHazardEngine
    from hazards.categories import HazardClassifier
    detector = HazardDetector()
    compound_engine = CompoundHazardEngine()
    classifier = HazardClassifier()
    all_hazards = {}

    for i, district in enumerate(districts):
        logger.info(f"  [{i+1}/{len(districts)}] {district}...")

        # ERA5 daily fetch (original notebook style)
        era5_data = {}
        if era5_loader:
            for var in ALL_ERA5_VARS:
                s = _fetch_era5_district(era5_loader, district, var)
                if not s.empty:
                    era5_data[var] = s
                    logger.info(f"    ERA5 {var}: {len(s)} daily records")

        # IMD fetch
        imd_data = {}
        if imd_loader:
            for var in ALL_IMD_VARS:
                s = _fetch_imd_district(bounds, imd_loader, district, var)
                if not s.empty:
                    imd_data[var] = s
                    logger.info(f"    IMD {var}: {len(s)} daily records")

        # Merge all sources
        combined = _merge_district_data(era5_data, imd_data, sat_data, district)
        if combined.empty:
            logger.warning(f"    No data for {district}, skipping")
            continue

        # Create unified precip/tmax/tmin columns — prefer ERA5 (covers to 2026), fill gaps with IMD
        for var in ["tmax","tmin","precip"]:
            if f"{var}_era5" in combined.columns:
                combined[var] = combined[f"{var}_era5"]
                if f"{var}_imd" in combined.columns:
                    combined[var] = combined[var].fillna(combined[f"{var}_imd"])
            elif f"{var}_imd" in combined.columns:
                combined[var] = combined[f"{var}_imd"]

        # Hazard detection
        lcc = detector.compute_climate_indices(combined)
        hazards = detector.detect_all(lcc)
        hazards["district"] = district

        # Compound hazards are already included in hazards (from compound.compute)

        for h in ["flood", "drought", "heatwave", "agri_stress"]:
            col = f"{h}_severity"
            if col in hazards.columns:
                hazards[f"{h}_class"] = hazards[col].apply(classifier.classify)

        all_hazards[district] = hazards
        out = Path(f"exports/hazards/{district}_hazards.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        hazards.to_csv(out, index=False)
        logger.info(f"    → {len(hazards)} hazard rows saved")

        # Also save combined raw data
        raw_out = Path(f"exports/raw/{district}_data.csv")
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(raw_out, index=False)
        logger.info(f"    → {len(combined)} raw data rows saved")

    # --- Step 7: ML ---
    if not args.skip_ml:
        logger.info("[5/8] Training ML models...")
        from models.classical import ClassicalModels
        from models.evaluator import ModelEvaluator
        ml = ClassicalModels()
        evaluator = ModelEvaluator()
        all_metrics = {}
        for district in all_hazards:
            hazards = all_hazards[district]
            logger.info(f"  Training on {district} ({len(hazards)} samples)...")
            try:
                features = [c for c in hazards.columns if c.endswith("_severity") or c in ["spi3", "vhi", "cdd"]]
                if len(features) < 3:
                    continue
                X = hazards[features].fillna(0).values
                y = (hazards[["flood_event", "drought_event", "heatwave_event"]].fillna(0).sum(axis=1) > 0).astype(int).values
                if len(np.unique(y)) < 2:
                    logger.warning(f"    Only one class ({y[0]}), skipping ML")
                    continue
                split = int(len(X) * 0.8)
                models = ml.train_all(X[:split], y[:split], task="classification")
                if models:
                    all_preds = np.zeros(len(X[split:]))
                    for name, model in models.items():
                        all_preds += model.predict(X[split:])
                    y_pred = (all_preds / len(models) > 0.5).astype(int)
                    metrics = evaluator.evaluate_classification(y_pred, y[split:])
                    all_metrics[district] = metrics
                    logger.info(f"    Accuracy: {metrics.get('accuracy', 0):.3f}")
            except Exception as e:
                logger.warning(f"    ML failed: {e}")
        if all_metrics:
            pd.DataFrame(all_metrics).T.to_csv("exports/ml_metrics.csv")
    else:
        logger.info("[5/8] ML skipped")

    # --- Step 8: Forecast (CMIP6 + ML blended) ---
    logger.info("[6/8] Generating forecasts (CMIP6 + ML)...")
    from forecasting.daily_forecast import DailyForecastEngine
    fc_engine = DailyForecastEngine()
    cache_path = Path(CFG.cache_dir) / "cmip6_ensemble.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            fc_engine.set_ensemble(cached)
            logger.info(f"  Loaded CMIP6 ensemble from cache ({len(cached)} rows)")
        except Exception as e:
            logger.warning(f"  CMIP6 cache load failed: {e}")
    elif cmip6_loader is not None and init_gee_result:
        logger.info("  Fetching CMIP6 projections from GEE...")
        try:
            raw = cmip6_loader.fetch_all_models()
            if not raw.empty:
                ensemble = cmip6_loader.compute_ensemble(raw)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                ensemble.to_parquet(cache_path, index=False)
                fc_engine.set_ensemble(ensemble)
                logger.info(f"  CMIP6 ensemble cached ({len(ensemble)} rows, {len(raw)} raw)")
        except Exception as e:
            logger.warning(f"  CMIP6 fetch failed: {e}")
    else:
        logger.info("  No CMIP6 data — using climatology fallback")
    for district in all_hazards:
        try:
            district_hazards = all_hazards[district]
            if "date" in district_hazards.columns:
                panel = district_hazards.set_index("date")
            else:
                panel = district_hazards
            for target in ["tmax", "precip"]:
                if target not in panel.columns and f"{target}_era5" in panel.columns:
                    target = f"{target}_era5"
                if target not in panel.columns:
                    continue
                fc_df = fc_engine.generate_daily_to_2040(panel, target, district)
                if fc_df is not None and not fc_df.empty:
                    out = Path(f"exports/forecasts/{district}_{target}_forecast.csv")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fc_df.to_csv(out, index=False)
        except Exception as e:
            logger.warning(f"  Forecast failed for {district}: {e}")

    # --- Step 9: Summary ---
    logger.info("[7/8] Exporting summary...")
    summary = {
        "start": t_start.isoformat(),
        "end": datetime.now().isoformat(),
        "districts": len(all_hazards),
        "districts_list": list(all_hazards.keys()),
        "hazard_rows": sum(len(h) for h in all_hazards.values()),
        "sources": {"era5": era5_loader is not None, "imd": imd_loader is not None, "satellite": not sat_data.empty},
    }
    (Path("exports") / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"  Summary → exports/pipeline_summary.json")

    elapsed = (datetime.now() - t_start).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s — {len(all_hazards)}/{len(districts)} districts")
    logger.info(f"  Sources: ERA5={'yes' if era5_loader else 'no'}, IMD={'yes' if imd_loader else 'no'}, Sat={'yes' if not sat_data.empty else 'no'}")
    logger.info(f"  Exports: D:\\cri\\exports\\ (hazards/, raw/, forecasts/)")
    logger.info("=" * 60)
    return summary


def main():
    args = parse_args()
    summary = run_pipeline(args)
    if args.dashboard:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        os.path.join(os.path.dirname(__file__), "dashboard", "app.py")])


if __name__ == "__main__":
    main()
