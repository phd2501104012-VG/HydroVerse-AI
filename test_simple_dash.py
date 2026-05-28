import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import numpy as np
from data.source_manager import DataSourceManager
from data.era5_loader import ERA5Loader
from data.imd_loader import IMDLoader
from geospatial.boundaries import DistrictBoundaries
from hazards.detection import HazardDetector
from hazards.categories import HazardClassifier

st.set_page_config(layout="wide")

@st.cache_resource
def init():
    b = DistrictBoundaries()
    ds = DataSourceManager()
    era5 = ERA5Loader()
    imd = IMDLoader()
    ds.set_era5_loader(era5)
    ds.set_imd_loader(imd)
    det = HazardDetector()
    cls = HazardClassifier()
    return b, ds, det, cls

bounds, ds_mgr, detector, classifier = init()
districts = bounds.district_names

st.title("HydroVerse AI - Test Dashboard")
district = st.selectbox("Select District", districts)

if district:
    with st.spinner(f"Loading {district}..."):
        data = ds_mgr.get_district_timeseries(district, ["tmax","tmin","precip"])
        if data is not None and not data.empty:
            if "date" in data.columns:
                data = data.set_index(pd.to_datetime(data["date"])).drop(columns=["date"])
            st.success(f"Loaded {len(data)} records")
            st.metric("Tmax", f"{data.tmax.mean():.1f}°C")
            st.metric("Tmin", f"{data.tmin.mean():.1f}°C")
            st.metric("Precip", f"{data.precip.mean():.1f}mm")
            st.line_chart(data.tail(365))
            hazards = detector.detect_all(data)
            for c in hazards.columns:
                if c.endswith("_severity"):
                    st.metric(c, f"{hazards[c].max():.1f}")
        else:
            st.error("No data returned!")
