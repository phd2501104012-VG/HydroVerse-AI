import pandas as pd, sys
sys.path.insert(0, "D:\\cri")
from pathlib import Path

d = "Sehore"
rp = Path("D:\\cri\\exports\\raw\\" + d + "_data.csv")
raw = pd.read_csv(rp)
print("Raw cols:", list(raw.columns))
hp = Path("D:\\cri\\exports\\hazards\\" + d + "_hazards.csv")
haz = pd.read_csv(hp)
print("Haz cols:", [c for c in haz.columns if "tmax" in c or "precip" in c or "ndvi" in c])

# Check for a few other "missing tmax" districts
for d2 in ["Seoni", "Shahdol", "Ujjain", "Vidisha"]:
    rp2 = Path("D:\\cri\\exports\\raw\\" + d2 + "_data.csv")
    h2 = pd.read_csv(rp2)
    cols = list(h2.columns)
    has_tmax_era5 = "tmax_era5" in cols
    print(f"{d2}: raw cols={len(cols)}, tmax_era5={has_tmax_era5}")
