import time
import hashlib
import re
from typing import List, Optional, Dict, Any
from functools import wraps

import pandas as pd
import numpy as np

from config import CFG
from utils.logger import log

try:
    import ee
    _HAS_EE = True
except ImportError:
    ee = None
    _HAS_EE = False


def init_gee(project: Optional[str] = None, verbose: bool = True) -> bool:
    if not _HAS_EE:
        if verbose:
            log.warning("GEE not available (install earthengine-api)")
        return False
    project = project or CFG.gee.project
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        _ = ee.Number(1).getInfo()
        if verbose:
            log.info(f"GEE authenticated (project: {project or 'default'})")
        return True
    except Exception as e:
        if verbose:
            log.error(f"GEE init failed: {e}")
        return False


def retry(max_attempts=3, backoff=2.0):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            wait = 1.0
            last = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    if attempt == max_attempts:
                        log.warning(f"{fn.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    log.info(f"{fn.__name__} attempt {attempt} failed; retry in {wait:.1f}s")
                    time.sleep(wait)
                    wait *= backoff
            raise last
        return wrapper
    return deco


def cache_key(*parts):
    raw = "_".join(str(p) for p in parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    if len(safe) > 180:
        h = hashlib.md5(raw.encode()).hexdigest()[:10]
        safe = safe[:160] + "_" + h
    return safe


def shapely_to_ee(geom):
    if not _HAS_EE:
        return None
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon([list(geom.exterior.coords)])
    if geom.geom_type == "MultiPolygon":
        coords = [[list(p.exterior.coords)] for p in geom.geoms]
        return ee.Geometry.MultiPolygon(coords)
    return None


def build_reducer(bands: List[str]):
    if not _HAS_EE:
        return None
    if len(bands) == 1:
        return ee.Reducer.mean().setOutputs([bands[0]])
    reducer = ee.Reducer.mean().setOutputs([bands[0]])
    for b in bands[1:]:
        reducer = reducer.combine(
            ee.Reducer.mean().setOutputs([b]),
            sharedInputs=False,
        )
    return reducer


@retry(max_attempts=4, backoff=2.0)
def fetch_collection_batch(fc, coll_id, bands, start, end, scale=None):
    if not _HAS_EE:
        return pd.DataFrame(columns=["district", "date"] + bands)
    scale = scale or CFG.gee.scale_m
    coll = (
        ee.ImageCollection(coll_id)
        .filterDate(start, end)
        .filterBounds(fc)
        .select(bands)
    )
    if coll.size().getInfo() == 0:
        return pd.DataFrame(columns=["district", "date"] + bands)

    reducer = build_reducer(bands)

    def per_image(img):
        d = img.date().format("YYYY-MM-dd")
        stats = img.reduceRegions(
            collection=fc, reducer=reducer, scale=scale,
        )
        return stats.map(lambda f: f.set({"date": d}).setGeometry(None))

    flat = coll.map(per_image).flatten()
    info = flat.getInfo()
    feats = info.get("features", [])

    rows = []
    for f in feats:
        props = f.get("properties", {})
        row = {"district": props.get("district"), "date": props.get("date")}
        for b in bands:
            row[b] = props.get(b)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        for b in bands:
            df[b] = pd.to_numeric(df[b], errors="coerce")
    return df


def _safe_chunk_days(fc) -> int:
    """Dynamically compute chunk size to stay under GEE's 5000-element limit.
    Larger chunks = fewer calls = much faster."""
    try:
        n_districts = fc.size().getInfo()
    except Exception:
        n_districts = 5
    if n_districts <= 0:
        n_districts = 1
    max_chunk = 5000 // n_districts
    return min(max_chunk, 365)  # at most yearly, adjusted for district count


def chunked_fetch(fc, coll_id, bands, start, end, chunk_days=None):
    if not _HAS_EE:
        return pd.DataFrame(columns=["district", "date"] + bands)
    chunk_days = chunk_days or _safe_chunk_days(fc)
    chunks = []
    cur = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_dt + pd.Timedelta(days=1))
        log.info(f"  chunk {cur.date()} -> {nxt.date()} ({chunk_days}d)")
        df = fetch_collection_batch(fc, coll_id, bands,
                                    cur.strftime("%Y-%m-%d"),
                                    nxt.strftime("%Y-%m-%d"))
        if not df.empty:
            chunks.append(df)
        cur = nxt
    if not chunks:
        return pd.DataFrame(columns=["district", "date"] + bands)
    return (pd.concat(chunks, ignore_index=True)
            .drop_duplicates(["district", "date"])
            .sort_values(["district", "date"]))


def cmip6_chunked_fetch(fc, model, scenario, bands, start, end, scale=25000):
    if not _HAS_EE:
        return pd.DataFrame(columns=["district", "date"] + bands)
    @retry(max_attempts=3, backoff=3.0)
    def _fetch():
        coll = (
            ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("model", model))
            .filter(ee.Filter.eq("scenario", scenario))
            .filterDate(start, end)
            .select(bands)
        )
        if coll.size().getInfo() == 0:
            return pd.DataFrame(columns=["district", "date"] + bands)
        return _cmip6_coll_to_df(coll, fc, bands, scale)
    return _fetch()


def cmip6_multi_model_fetch(fc, models, scenario, bands, start, end, scale=25000):
    """Fetch CMIP6 data for MULTIPLE models in a single GEE query (~8x faster)."""
    if not _HAS_EE:
        return pd.DataFrame(columns=["district", "date", "model"] + bands)
    @retry(max_attempts=3, backoff=3.0)
    def _fetch():
        coll = (
            ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.inList("model", models))
            .filter(ee.Filter.eq("scenario", scenario))
            .filterDate(start, end)
            .select(bands)
        )
        if coll.size().getInfo() == 0:
            return pd.DataFrame(columns=["district", "date", "model"] + bands)
        return _cmip6_coll_to_df(coll, fc, bands, scale, add_model=True)
    return _fetch()


def _cmip6_coll_to_df(coll, fc, bands, scale=25000, add_model=False):
    """Execute a CMIP6 ImageCollection query and return a DataFrame."""
    reducer = build_reducer(bands)

    def per_image(img):
        d = img.date().format("YYYY-MM-dd")
        props = {"date": d}
        if add_model:
            props["model"] = img.get("model")
        stats = img.reduceRegions(
            collection=fc, reducer=reducer, scale=scale,
        )
        return stats.map(lambda f: f.set(props).setGeometry(None))

    flat = coll.map(per_image).flatten()
    info = flat.getInfo()
    feats = info.get("features", [])

    rows = []
    for f in feats:
        props = f.get("properties", {})
        row = {"district": props.get("district"), "date": props.get("date")}
        if add_model:
            row["model"] = props.get("model")
        for b in bands:
            row[b] = props.get(b)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        for b in bands:
            df[b] = pd.to_numeric(df[b], errors="coerce")
    return df
