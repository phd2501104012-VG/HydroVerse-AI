"""Quick check: flood & drought from CMIP6 data alone"""
import sys; sys.path.insert(0, r"D:\cri")
import pandas as pd
import numpy as np
from pathlib import Path
from config import CFG
from hazards.drought import DroughtDetector
from hazards.flood import FloodDetector
from hazards.categories import HazardClassifier

cache_dir = Path(CFG.cache_dir)
district = "Agar Malwa"

# Load CMIP6 ensemble and extract precip/tmax for this district
ens = pd.read_parquet(cache_dir / "cmip6_ensemble.parquet")
sub = ens[ens["district"] == district].copy()
sub["date"] = pd.to_datetime(sub["date"])
sub = sub.set_index("date").sort_index()

precip = pd.to_numeric(sub["precip_proj_mean"], errors="coerce")
tmax = pd.to_numeric(sub["tmax_proj_mean"], errors="coerce")

# Load ERA5 historical data (2000-2026) to combine with CMIP6 for SPI
era5_file = list(cache_dir.glob("era5_precip_*.parquet"))[0]
era5 = pd.read_parquet(era5_file)
era5_sub = era5[era5["district"] == district][["date","total_precipitation_sum"]].copy()
era5_sub["date"] = pd.to_datetime(era5_sub["date"])
era5_sub = era5_sub.set_index("date") * 1000  # m to mm
combined_precip = pd.concat([era5_sub["total_precipitation_sum"], precip])
combined_precip.name = "precip"

# Also combine tmax
era5_tmax = list(cache_dir.glob("era5_tmax_*.parquet"))[0]
era5_t = pd.read_parquet(era5_tmax)
era5_t = era5_t[era5_t["district"] == district][["date","temperature_2m_max"]].copy()
era5_t["date"] = pd.to_datetime(era5_t["date"])
era5_t = era5_t.set_index("date") - 273.15
combined_tmax = pd.concat([era5_t["temperature_2m_max"], tmax])
combined_tmax.name = "tmax"

# Build combined DataFrame
df = pd.DataFrame({"precip": combined_precip, "tmax": combined_tmax}).sort_index()

print(f"Combined data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")

# Separate ERA5 period vs CMIP6 period
era5_end = era5_sub.index.max()
cmip6_start = precip.index.min()

# ── FLOOD ANALYSIS ──
print("\n=== FLOOD ===")
IMD_HEAVY = CFG.hazard.imd_precip_heavy  # 64.5mm
IMD_VERY_HEAVY = CFG.hazard.imd_precip_very_heavy  # 115.6mm

# ERA5 period
p_hist = df.loc[:era5_end, "precip"]
heavy_hist = (p_hist >= IMD_HEAVY).sum()
vheavy_hist = (p_hist >= IMD_VERY_HEAVY).sum()
print(f"ERA5 ({p_hist.index.min().date()} to {p_hist.index.max().date()}): "
      f"heavy_rain={heavy_hist}/{len(p_hist)} ({100*heavy_hist/len(p_hist):.2f}%), "
      f"very_heavy={vheavy_hist}")

# CMIP6 period  
p_cmip6 = df.loc[cmip6_start:, "precip"]
heavy_cmip6 = (p_cmip6 >= IMD_HEAVY).sum()
vheavy_cmip6 = (p_cmip6 >= IMD_VERY_HEAVY).sum()
print(f"CMIP6 ({p_cmip6.index.min().date()} to {p_cmip6.index.max().date()}): "
      f"heavy_rain={heavy_cmip6}/{len(p_cmip6)} ({100*heavy_cmip6/len(p_cmip6):.2f}%), "
      f"very_heavy={vheavy_cmip6}")

# Annual precip totals for CMIP6 period
print("\nAnnual precip totals (CMIP6):")
for year in range(2026, 2041):
    yr = p_cmip6[p_cmip6.index.year == year]
    if len(yr) > 0:
        total = yr.sum()
        heavy = (yr >= IMD_HEAVY).sum()
        print(f"  {year}: {total:.0f}mm, heavy_days={heavy}")

# ── DROUGHT ANALYSIS ──
print("\n=== DROUGHT (SPI-3m) ===")
dd = DroughtDetector()
spi3 = dd.compute_spi(df["precip"], scale_months=3)

# SPI for CMIP6 period
spi_cmip6 = spi3.loc[cmip6_start:]
print(f"SPI-3m during CMIP6 period:")
print(f"  min={spi_cmip6.min():.2f}, max={spi_cmip6.max():.2f}, mean={spi_cmip6.mean():.2f}")
print(f"  Drought (SPI<-1): {(spi_cmip6 < -1).sum()}/{len(spi_cmip6)} days")
print(f"  Severe drought (SPI<-1.5): {(spi_cmip6 < -1.5).sum()}/{len(spi_cmip6)} days")
print(f"  Extreme drought (SPI<-2): {(spi_cmip6 < -2).sum()}/{len(spi_cmip6)} days")

# Monthly SPI values
print("\nMonthly SPI-3m (last value each month):")
spi_monthly = spi_cmip6.resample("ME").last()
for d, v in spi_monthly.items():
    cat = "Normal"
    if v < -2: cat = "Extreme Drought"
    elif v < -1.5: cat = "Severe Drought"
    elif v < -1: cat = "Moderate Drought"
    elif v < 0: cat = "Mild Drought"
    print(f"  {d.date()}: SPI={v:.2f} ({cat})")

# ── HEATWAVE ANALYSIS ──
print("\n=== HEATWAVE (tmax) ===")
# Check how many days exceed 40°C in CMIP6 period
t_cmip6 = df.loc[cmip6_start:, "tmax"]
hot_days = (t_cmip6 >= 40).sum()
very_hot = (t_cmip6 >= 45).sum()
print(f"Days tmax>=40°C: {hot_days}/{len(t_cmip6)} ({100*hot_days/len(t_cmip6):.1f}%)")
print(f"Days tmax>=45°C: {very_hot}/{len(t_cmip6)}")
print(f"Max tmax: {t_cmip6.max():.1f}°C")

# Monthly max temperatures
print("\nMonthly max tmax:")
for year in range(2026, 2041):
    yr = t_cmip6[t_cmip6.index.year == year]
    if len(yr) > 0:
        print(f"  {year}: max={yr.max():.1f}°C, days>=40°C={(yr>=40).sum()}")
