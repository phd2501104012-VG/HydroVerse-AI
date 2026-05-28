"""Fetch missing ERA5 tmin for all 51 districts and cache it."""
import sys, os
sys.path.insert(0, "D:\\cri")
from pathlib import Path
from config import CFG
from utils.gee_utils import init_gee, chunked_fetch

init_gee(project=CFG.gee.project)

from geospatial.boundaries import DistrictBoundaries
bounds = DistrictBoundaries()
districts = bounds.district_names
print(f"Loading boundaries for {len(districts)} districts")

ee_fc = bounds.get_ee_fc(districts)
if ee_fc is None:
    print("ERROR: No FeatureCollection")
    sys.exit(1)

from data.era5_loader import ERA5Loader
loader = ERA5Loader(ee_fc)

# Force fresh cache for tmin only
print("Fetching ERA5 tmin for all 51 districts...")
df = loader.fetch_variable("tmin", force_refresh=True)
print(f"Done: {len(df)} records, {df['district'].nunique()} districts")
