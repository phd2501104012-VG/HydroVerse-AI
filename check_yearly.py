"""Check if CMIP6 patterns repeat identically year after year."""
import sys; sys.path.insert(0, r"D:\cri")
import pandas as pd
import numpy as np
from pathlib import Path
from config import CFG

ens = pd.read_parquet(Path(CFG.cache_dir) / "cmip6_ensemble.parquet")
dist = "Agar Malwa"
sub = ens[ens["district"] == dist].copy()
sub["date"] = pd.to_datetime(sub["date"])
sub = sub.set_index("date").sort_index()

# Pick a specific day-of-year across all years and check values
print("=== Same DOY across different years ===")
for doy in [135, 150, 200, 250, 300]:  # May 15, May 30, Jul 19, Sep 7, Oct 27
    print(f"\nDOY {doy}:")
    for year in [2026, 2028, 2030, 2035, 2040]:
        d = f"{year}-{pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=doy-1):%m-%d}"
        if d in sub.index:
            row = sub.loc[d]
            tmax = float(row["tmax_proj_mean"])
            precip = float(row["precip_proj_mean"])
            print(f"  {d}: tmax={tmax:.1f}, precip={precip:.3f}")

# Check if all May 15 values are the same
print("\n=== All May 15 values (2026-2040) ===")
may15_vals = []
for year in range(2026, 2041):
    d = f"{year}-05-15"
    if d in sub.index:
        row = sub.loc[d]
        may15_vals.append(float(row["tmax_proj_mean"]))
print(f"  May 15 tmax: {may15_vals}")
print(f"  Unique values: {len(set(round(v,2) for v in may15_vals))}")

# Check if the ENTIRE time series is just the same year repeated
print("\n=== Year-by-year comparison ===")
for year in [2026, 2027, 2028]:
    yr = sub[sub.index.year == year]
    print(f"\n{year}: tmax range={yr['tmax_proj_mean'].min():.1f}-{yr['tmax_proj_mean'].max():.1f}, "
          f"precip range={yr['precip_proj_mean'].min():.3f}-{yr['precip_proj_mean'].max():.3f}, "
          f"precip sum={yr['precip_proj_mean'].sum():.1f}")

# Check if 2026 and 2027 are identical by day-of-year alignment
y26 = sub[sub.index.year == 2026].copy()
y27 = sub[sub.index.year == 2027].copy()
y26_doy = y26.set_index(y26.index.dayofyear)
y27_doy = y27.set_index(y27.index.dayofyear)
diff = (y26_doy["tmax_proj_mean"] - y27_doy["tmax_proj_mean"]).abs()
print(f"\n=== 2026 vs 2027 DOY alignment ===")
print(f"  tmax mean abs diff: {diff.mean():.4f}°C")
print(f"  tmax max abs diff: {diff.max():.4f}°C")
print(f"  Identical DOY (diff<0.01): {(diff < 0.01).sum()}/{len(diff)} days")

diff_p = (y26_doy["precip_proj_mean"] - y27_doy["precip_proj_mean"]).abs()
print(f"  precip mean abs diff: {diff_p.mean():.4f}mm")
print(f"  precip max abs diff: {diff_p.max():.4f}mm")
