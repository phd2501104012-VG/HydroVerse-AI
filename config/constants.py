RISK_CLASSES = ["Normal", "Watch", "Warning", "Severe", "Extreme"]
RISK_THRESHOLDS = [0, 25, 50, 75, 90]
RISK_COLORS = {
    "Normal": "#22c55e",
    "Watch": "#eab308",
    "Warning": "#f97316",
    "Severe": "#ef4444",
    "Extreme": "#7f1d1d",
}
RISK_COLORS_HEX = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#7f1d1d"]

HAZARD_COLORS = {
    "flood": "#2563eb",
    "drought": "#ca8a04",
    "heatwave": "#dc2626",
    "agri_stress": "#16a34a",
    "extreme_precip": "#8b5cf6",
    "compound": "#ec4899",
}

HAZARD_ICONS = {
    "flood": "🌊",
    "drought": "🏜️",
    "heatwave": "🔥",
    "agri_stress": "🌾",
    "extreme_precip": "⛈️",
    "compound": "⚠️",
}

CMIP6_VARIABLES = {
    "tas": {"band": "tas", "alias": "tmean_proj", "scale": 1.0, "offset": -273.15, "unit": "C"},
    "tasmax": {"band": "tasmax", "alias": "tmax_proj", "scale": 1.0, "offset": -273.15, "unit": "C"},
    "tasmin": {"band": "tasmin", "alias": "tmin_proj", "scale": 1.0, "offset": -273.15, "unit": "C"},
    "pr": {"band": "pr", "alias": "precip_proj", "scale": 86400.0, "offset": 0, "unit": "mm"},
    "hurs": {"band": "hurs", "alias": "rh_proj", "scale": 1.0, "offset": 0, "unit": "%"},
    "sfcWind": {"band": "sfcWind", "alias": "wind_proj", "scale": 1.0, "offset": 0, "unit": "m/s"},
    "rsds": {"band": "rsds", "alias": "solar_proj", "scale": 1.0, "offset": 0, "unit": "W/m²"},
}

ERA5_DATASETS = {
    "tmax": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "temperature_2m_max", "scale": 1, "offset": -273.15, "unit": "°C", "valid_range": (-20, 60)},
    "tmin": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "temperature_2m_min", "scale": 1, "offset": -273.15, "unit": "°C", "valid_range": (-30, 45)},
    "tmean": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "temperature_2m", "scale": 1, "offset": -273.15, "unit": "°C", "valid_range": (-30, 55)},
    "precip": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "total_precipitation_sum", "scale": 1000, "offset": 0, "unit": "mm", "valid_range": (0, 1500)},
    "soil_moisture": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "volumetric_soil_water_layer_1", "scale": 1, "offset": 0, "unit": "m³/m³", "valid_range": (0, 0.6)},
}

REALTIME_DATASETS = {
    "precip": {"collection": "NASA/GPM_L3/IMERG_V06", "band": "precipitationCal", "scale": 1, "offset": 0, "unit": "mm/hr"},
    "soil_moisture": {"collection": "ECMWF/ERA5_LAND/DAILY_AGGR", "band": "volumetric_soil_water_layer_1", "scale": 1, "offset": 0, "unit": "m³/m³"},
    "ndvi": {"collection": "MODIS/061/MOD13Q1", "band": "NDVI", "scale": 0.0001, "offset": 0, "unit": "index"},
    "lst": {"collection": "MODIS/061/MOD11A1", "band": "LST_Day_1km", "scale": 0.02, "offset": 0, "unit": "K"},
    "lst_night": {"collection": "MODIS/061/MOD11A1", "band": "LST_Night_1km", "scale": 0.02, "offset": 0, "unit": "K"},
}

ACTION_PLAYBOOK = {
    "flood": {
        "Normal": "No action required.",
        "Watch": "Monitor reservoir levels; check drainage in low-lying areas.",
        "Warning": "Issue flood advisory; pre-position rescue teams.",
        "Severe": "Activate flood control room; evacuate riverside hamlets.",
        "Extreme": "Declare emergency; coordinate with SDRF/NDRF.",
    },
    "drought": {
        "Normal": "No action required.",
        "Watch": "Review reservoir storage; notify Agriculture dept.",
        "Warning": "Implement water rationing for non-essential uses.",
        "Severe": "Crop loss assessment; prepare fodder banks.",
        "Extreme": "Declare drought; release relief funds; emergency water tankering.",
    },
    "heatwave": {
        "Normal": "No action required.",
        "Watch": "Issue heat advisory; remind public of hydration.",
        "Warning": "Adjust school/outdoor work hours; open cooling centers.",
        "Severe": "Halt outdoor labor 11am-4pm; ambulance pre-positioning.",
        "Extreme": "Declare emergency; mandatory school closure; hospital surge.",
    },
    "agri_stress": {
        "Normal": "No action required.",
        "Watch": "KVK advisory on irrigation timing.",
        "Warning": "Soil moisture monitoring; recommend protective irrigation.",
        "Severe": "Crop insurance pre-assessment; advance MNREGA work plans.",
        "Extreme": "Declare crop loss; activate PMFBY claims.",
    },
    "compound": {
        "Normal": "No action required.",
        "Watch": "Multi-hazard monitoring activated.",
        "Warning": "Inter-agency coordination initiated.",
        "Severe": "Emergency operations center activated.",
        "Extreme": "Full disaster response mobilization.",
    },
}

DASHBOARD_BRANDING = {
    "HydroVerse AI": {
        "tagline": "AI-Powered Real-Time Climate Hazard Intelligence Platform",
        "primary": "#00d4ff",
        "secondary": "#7c3aed",
        "accent": "#f59e0b",
    },
    "Climatrix AI": {
        "tagline": "Next-Generation Climate Risk Intelligence",
        "primary": "#06b6d4",
        "secondary": "#8b5cf6",
        "accent": "#f97316",
    },
    "GeoHazard Nexus": {
        "tagline": "Geospatial Hazard Intelligence Platform",
        "primary": "#10b981",
        "secondary": "#3b82f6",
        "accent": "#ef4444",
    },
}
