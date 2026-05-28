"""Fetch remaining 7 CMIP6 models from GEE and rebuild ensemble."""
import sys; sys.path.insert(0, r"D:\cri")
import pandas as pd
import numpy as np
from pathlib import Path
from config import CFG, constants
from data.cmip6_loader import CMIP6Loader
from geospatial.boundaries import DistrictBoundaries
from utils.gee_utils import init_gee
from utils.logger import log

init_gee()
cache_dir = Path(CFG.cache_dir)

# 1. Get EE feature collection
bounds = DistrictBoundaries()
ee_fc = bounds.get_ee_fc()
if ee_fc is None:
    log.error("Failed to create EE feature collection")
    sys.exit(1)

# 2. Create loader
loader = CMIP6Loader(ee_fc)
models = ["ACCESS-CM2", "CanESM5", "MIROC6", "MPI-ESM1-2-LR",
          "IPSL-CM6A-LR", "CNRM-CM6-1", "UKESM1-0-LL", "MRI-ESM2-0"]
scenario = CFG.forecasting.ssp_scenario
start_year = CFG.forecasting.future_start_year
end_year = CFG.forecasting.future_end_year
variables = ["tasmax", "tasmin", "pr"]

# 3. Check which models already fully cached
existing_models = set()
for p in cache_dir.glob("cmip6_*_*.parquet"):
    parts = p.stem.split("_")
    if len(parts) >= 2:
        existing_models.add(parts[1])

# Check which years per model
def model_years_cached(model):
    years = set()
    for y in range(start_year, end_year + 1):
        if (cache_dir / f"cmip6_{model}_{scenario}_{y}.parquet").exists():
            years.add(y)
    return years

to_fetch = []
for m in models:
    cached_years = model_years_cached(m)
    expected_years = set(range(start_year, end_year + 1))
    missing = expected_years - cached_years
    if missing:
        to_fetch.append((m, missing))
        log.info(f"  {m}: {len(cached_years)}/{len(expected_years)} years cached, missing {sorted(missing)}")
    else:
        log.info(f"  {m}: fully cached ({len(cached_years)} years)")

# 4. Fetch missing models/years
for model, missing_years in to_fetch:
    for year in sorted(missing_years):
        log.info(f"  Fetching {model} {year}...")
        try:
            df = loader.fetch_model_year(model, scenario, year, variables=variables, month_by_month=False)
            if not df.empty:
                fp = cache_dir / f"cmip6_{model}_{scenario}_{year}.parquet"
                df.to_parquet(fp, index=False)
                log.info(f"    Saved {fp.name} ({len(df)} rows)")
            else:
                log.warning(f"    Empty result for {model} {year}")
        except Exception as e:
            log.warning(f"    Failed {model} {year}: {e}")

# 5. Rebuild ensemble from ALL cached model data
log.info("Rebuilding ensemble from all cached model data...")
all_raw = []
for model in models:
    for year in range(start_year, end_year + 1):
        fp = cache_dir / f"cmip6_{model}_{scenario}_{year}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            all_raw.append(df)

if not all_raw:
    log.error("No model data found!")
    sys.exit(1)

raw = pd.concat(all_raw, ignore_index=True)
log.info(f"Total raw rows: {len(raw)} from {raw['model'].nunique() if 'model' in raw.columns else '?'} models")

# 6. Compute ensemble
ensemble = loader.compute_ensemble(raw)
ensemble_fp = cache_dir / "cmip6_ensemble.parquet"
ensemble.to_parquet(ensemble_fp, index=False)
log.info(f"Ensemble saved: {ensemble_fp} ({len(ensemble)} rows)")

# 7. Print model stats
if "model" in raw.columns:
    print(f"\n=== Model counts ===")
    for m in raw["model"].value_counts().sort_index().items():
        print(f"  {m[0]}: {m[1]} rows")
print(f"\nEnsemble columns: {ensemble.columns.tolist()}")
print(f"Ensemble date range: {ensemble['date'].min()} to {ensemble['date'].max()}")
