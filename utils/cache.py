import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config import CFG
from utils.logger import log


class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or CFG.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str, ext: str = "pkl") -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe}.{ext}"

    def get(self, key: str, max_age_hours: Optional[float] = None) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        if max_age_hours is not None:
            age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
            if age > max_age_hours:
                return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any):
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def get_dataframe(self, key: str, max_age_hours: Optional[float] = None) -> Optional[pd.DataFrame]:
        path = self._key_to_path(key, "parquet")
        if not path.exists():
            return None
        if max_age_hours is not None:
            age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
            if age > max_age_hours:
                return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def set_dataframe(self, key: str, df: pd.DataFrame):
        path = self._key_to_path(key, "parquet")
        df.to_parquet(path, index=False)

    def clear(self, older_than_hours: Optional[float] = None):
        for f in self.cache_dir.iterdir():
            if f.is_file():
                if older_than_hours is not None:
                    age = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).total_seconds() / 3600
                    if age > older_than_hours:
                        f.unlink()
                else:
                    f.unlink()
        log.info(f"Cache cleared ({self.cache_dir})")


cache_manager = CacheManager()
