import pandas as pd, sys, numpy as np
sys.path.insert(0, "D:\\cri")
from pathlib import Path
from forecasting.daily_forecast import DailyForecastEngine

# Test a district missing tmax forecast
d = "Sehore"
hp = Path("D:\\cri\\exports\\hazards\\" + d + "_hazards.csv")
haz = pd.read_csv(hp)
if "date" in haz.columns:
    haz = haz.set_index(pd.to_datetime(haz["date"]))
else:
    haz.index = pd.RangeIndex(len(haz))

print("Haz cols with tmax:", [c for c in haz.columns if "tmax" in c])
print("Haz cols with precip:", [c for c in haz.columns if "precip" in c])

# Check if tmax exists
if "tmax" in haz.columns:
    print("tmax in columns:", haz["tmax"].notna().sum())
if "tmax_era5" in haz.columns:
    print("tmax_era5 in columns:", haz["tmax_era5"].notna().sum())

# Try generating forecast
fc_engine = DailyForecastEngine()
try:
    fc = fc_engine.generate_daily_to_2040(haz, "tmax_era5", d)
    print("Forecast result:", type(fc), "empty:", fc.empty if fc is not None else "None")
    if fc is not None and not fc.empty:
        print("Forecast range:", fc["date"].min(), "to", fc["date"].max())
except Exception as e:
    print(f"Forecast failed: {e}")
