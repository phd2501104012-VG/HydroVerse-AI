from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum


class SystemMode(Enum):
    OPERATIONAL = "operational"
    RESEARCH = "research"
    VALIDATION = "validation"
    TRAINING = "training"


class DataSource(Enum):
    ERA5 = "ERA5"
    IMD = "IMD"
    CHIRPS = "CHIRPS"
    CMIP6 = "CMIP6"
    AUTO = "Auto"


class HazardType(Enum):
    FLOOD = "flood"
    DROUGHT = "drought"
    HEATWAVE = "heatwave"
    AGRI_STRESS = "agri_stress"
    EXTREME_PRECIP = "extreme_precip"
    COMPOUND = "compound"


class RiskLevel(Enum):
    NORMAL = "Normal"
    WATCH = "Watch"
    WARNING = "Warning"
    SEVERE = "Severe"
    EXTREME = "Extreme"


@dataclass
class IMDConfig:
    precip_dir: str = r"D:\Data\IMD_0.05\INDmet_Netcdf_Data\INDmet_Netcdf_Data\Yearly_File_Precipitation"
    tmax_dir: str = r"D:\Data\IMD_0.05\INDmet_Netcdf_Data\INDmet_Netcdf_Data\Yearly_File_Tmax"
    tmin_dir: str = r"D:\Data\IMD_0.05\INDmet_Netcdf_Data\INDmet_Netcdf_Data\Yearly_File_Tmin"
    tmean_dir: str = r"D:\Data\IMD_0.05\INDmet_Netcdf_Data\INDmet_Netcdf_Data\Yearly_File_Tmean"
    precip_pattern: str = r"INDmet_precipitation_05km_*.nc"
    tmax_pattern: str = r"INDmet_tmax_05km_*.nc"
    tmin_pattern: str = r"INDmet_tmin_05km_*.nc"
    tmean_pattern: str = r"INDmet_tmean_05km_*.nc"
    start_year: int = 1981
    end_year: int = 2024
    chunk_size: str = "500MB"
    lat_name: str = "lat"
    lon_name: str = "lon"
    time_name: str = "time"
    precip_var: str = "precipitation"
    tmax_var: str = "tmax"
    tmin_var: str = "tmin"
    tmean_var: str = "tmean"


@dataclass
class GEEConfig:
    project: str = "floodmaping"
    scale_m: int = 5000
    chunk_days: int = 365
    cmip6_chunk_years: int = 3
    max_retries: int = 4
    retry_backoff: float = 2.0


@dataclass
class ForecastingConfig:
    ml_horizon_days: int = 90
    blend_days: int = 30
    future_start_year: int = 2026
    future_end_year: int = 2040
    ssp_scenario: str = "ssp245"
    cmip6_models: List[str] = field(default_factory=lambda: [
        "ACCESS-CM2", "CanESM5", "MIROC6", "MPI-ESM1-2-LR",
        "IPSL-CM6A-LR", "CNRM-CM6-1", "UKESM1-0-LL", "MRI-ESM2-0",
    ])
    ensemble_size: int = 100
    confidence_level: float = 0.95


@dataclass
class HazardConfig:
    imd_precip_light: float = 15.6
    imd_precip_moderate: float = 15.6
    imd_precip_heavy: float = 64.5
    imd_precip_very_heavy: float = 115.6
    imd_precip_extreme: float = 204.5
    heatwave_tmax_threshold: float = 40.0
    heatwave_departure: float = 4.5
    heatwave_severe_departure: float = 6.5
    drought_spi_threshold: float = -1.0
    drought_sustained_days: int = 30
    flood_persistence_days: int = 2
    agri_vhi_threshold: float = 50.0
    risk_thresholds: List[float] = field(default_factory=lambda: [0, 25, 50, 75, 90])


@dataclass
class MLConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_estimators: int = 300
    max_depth: int = 12
    learning_rate: float = 0.05
    lstm_units: int = 128
    lstm_layers: int = 2
    transformer_heads: int = 8
    transformer_layers: int = 4
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30, 60, 90])
    rolls: List[int] = field(default_factory=lambda: [7, 30, 90])
    cross_lags: List[int] = field(default_factory=lambda: [90, 180, 365])


@dataclass
class RealTimeConfig:
    update_frequency_minutes: int = 60
    history_days: int = 90
    anomaly_window: int = 30
    alert_cooldown_hours: int = 6
    max_alerts_per_district: int = 5


@dataclass
class DashboardConfig:
    title: str = "HydroVerse AI"
    subtitle: str = "AI-Powered Real-Time Climate Hazard Intelligence Platform"
    theme: str = "dark"
    map_zoom: int = 7
    map_center_lat: float = 23.5
    map_center_lon: float = 78.5
    port: int = 8501
    animation_fps: int = 10
    refresh_interval_ms: int = 60000


@dataclass
class Config:
    # State
    state_name: str = "Madhya Pradesh"
    district_col: str = "NAME_2"
    state_col: str = "NAME_1"

    # Shapefiles
    shp_candidates: List[str] = field(default_factory=lambda: [
        "./MP_SHAPE/MP_DISTRICT.shp",
        r"D:\cri (2)\cri\MP_SHAPE\MP_DISTRICT.shp",
        "./MP_SHAPE/MP_state.shp",
        r"D:\cri (2)\cri\MP_SHAPE\MP_state.shp",
        "./India_District_Boundary.shp",
        "./data/India_District_Boundary.shp",
        "./cache_climate/mp_districts.geojson",
    ])

    # Historical
    hist_start: str = "2000-01-01"
    hist_end: str = ""  # auto-set to today in __post_init__
    climatology_end: str = "2020-12-31"

    # Sub-configs
    imd: IMDConfig = field(default_factory=IMDConfig)
    gee: GEEConfig = field(default_factory=GEEConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Data source
    active_data_source: DataSource = DataSource.AUTO

    # Paths
    cache_dir: str = r"D:\cri\data\cache"
    export_dir: str = "./data/exports"
    realtime_dir: str = "./data/realtime"
    imd_dir: str = "./data/imd"

    # Mode
    mode: SystemMode = SystemMode.OPERATIONAL

    # All MP districts
    all_mp_districts: List[str] = field(default_factory=lambda: [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat",
        "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur",
        "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas",
        "Dhar", "Dindori", "Guna", "Gwalior", "Harda",
        "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni",
        "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena",
        "Narsinghpur", "Neemuch", "Niwari", "Panna", "Raisen",
        "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna",
        "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur",
        "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", "Ujjain",
        "Umaria", "Vidisha",
    ])

    # Target districts (can subset for prototype)
    target_districts: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_districts is None:
            self.target_districts = self.all_mp_districts
        if not self.hist_end:
            self.hist_end = datetime.now().strftime("%Y-%m-%d")


CFG = Config()
