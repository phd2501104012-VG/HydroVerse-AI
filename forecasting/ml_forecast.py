from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge

from config import CFG
from utils.logger import log


try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class MLForecastGenerator:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._feature_cols: Dict[str, List[str]] = {}
        self._best_models: Dict[tuple, Dict] = {}

    def build_features(self, panel: pd.DataFrame, target: str) -> tuple:
        df = pd.DataFrame(index=panel.index)
        df[target] = panel[target]
        lags = CFG.ml.lags
        rolls = CFG.ml.rolls
        cross_lags = CFG.ml.cross_lags

        for lag in lags:
            df[f"{target}_lag{lag}"] = panel[target].shift(lag)

        for w in rolls:
            df[f"{target}_roll{w}_mean"] = panel[target].shift(1).rolling(w, min_periods=max(1, w // 3)).mean()
            df[f"{target}_roll{w}_std"] = panel[target].shift(1).rolling(w, min_periods=max(1, w // 3)).std()

        cross_vars = [
            c for c in ["tmax", "tmin", "precip", "soil_moisture", "ndvi", "spi_3m"]
            if c in panel.columns and c != target
        ]
        for cv in cross_vars[:5]:
            for lag in cross_lags:
                df[f"{cv}_lag{lag}"] = panel[cv].shift(lag)

        d = df.index
        df["doy"] = d.dayofyear
        df["month"] = d.month
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
        df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

        train_end = pd.Timestamp(CFG.climatology_end)
        train = panel[target][panel.index <= train_end]
        if len(train) > 0:
            doy_clim = train.groupby(train.index.dayofyear).agg(["mean", "std"])
            df["target_doy_mean"] = df["doy"].map(doy_clim["mean"])
            df["target_doy_std"] = df["doy"].map(doy_clim["std"])

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        feat_cols = [c for c in df.columns if c != target]
        return df, feat_cols

    def train(
        self,
        panel: pd.DataFrame,
        target: str,
        district: str,
    ) -> Dict:
        df_feat, feat_cols = self.build_features(panel, target)
        if df_feat.empty or len(df_feat) < 200:
            return {}

        X = df_feat[feat_cols].values
        y = df_feat[target].values
        n = len(X)
        n_test = max(60, int(n * 0.2))
        split = n - n_test
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        scaler = StandardScaler().fit(X_tr)
        self._scalers[f"{district}_{target}"] = scaler
        self._feature_cols[f"{district}_{target}"] = feat_cols

        results = {}
        rs = CFG.ml.random_state

        def _try_model(name, model):
            try:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                results[name] = {
                    "model": model,
                    "pred": pred,
                    "y_test": y_te,
                    "dates_test": df_feat.index[split:],
                    "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                    "mae": float(mean_absolute_error(y_te, pred)),
                    "r2": float(r2_score(y_te, pred)),
                }
            except Exception as e:
                log.warning(f"{name} failed: {e}")

        _try_model("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=rs))
        _try_model("ExtraTrees", ExtraTreesRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=rs))
        _try_model("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=rs))
        _try_model("Ridge", Ridge(alpha=1.0))

        if HAS_XGB:
            _try_model("XGBoost", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbosity=0, n_jobs=-1))
        if HAS_LGB:
            _try_model("LightGBM", LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbose=-1, n_jobs=-1))

        if results:
            best = min(results.items(), key=lambda kv: kv[1]["rmse"])
            log.info(f"  Best {target}: {best[0]} (RMSE={best[1]['rmse']:.3f})")
            self._best_models[(district, target)] = best[1]

        return results

    def generate_forecast(
        self,
        panel: pd.DataFrame,
        target: str,
        district: str,
        horizon_days: int = 90,
    ) -> pd.DataFrame:
        key = f"{district}_{target}"
        if key not in self._scalers or key not in self._feature_cols:
            results = self.train(panel, target, district)
            if not results:
                return pd.DataFrame()

        best = self._best_models.get((district, target))
        if best is None:
            return pd.DataFrame()

        model = best["model"]
        feat_cols = self._feature_cols[key]
        last_date = panel.index.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        history = pd.Series(panel[target].dropna().values, index=panel[target].dropna().index)

        train = panel[target][panel.index <= pd.Timestamp(CFG.climatology_end)]
        doy_clim = train.groupby(train.index.dayofyear).agg(["mean", "std"])
        cross_vars = [c.split("_lag")[0] for c in feat_cols if "_lag" in c and c.split("_lag")[0] != target]
        cross_vars = list(set(cross_vars))

        forecasts = []
        rmse_hold = best.get("rmse", float(np.std(panel[target].dropna()) * 0.3))

        for fd in future_dates:
            row = {}
            for lag in CFG.ml.lags:
                row[f"{target}_lag{lag}"] = history.iloc[-lag] if lag <= len(history) else history.mean()

            for w in CFG.ml.rolls:
                recent = history.iloc[-w:] if w <= len(history) else history
                row[f"{target}_roll{w}_mean"] = recent.mean()
                row[f"{target}_roll{w}_std"] = recent.std() if len(recent) > 1 else 0

            for cv in cross_vars:
                for lag in CFG.ml.cross_lags:
                    lookup_date = fd - pd.Timedelta(days=lag)
                    col = f"{cv}_lag{lag}"
                    if col in feat_cols:
                        if lookup_date in panel.index:
                            val = panel.loc[lookup_date, cv] if cv in panel.columns else panel[target].mean()
                            row[col] = float(val) if pd.notna(val) else panel[target].mean()
                        else:
                            row[col] = panel[target].mean()

            row["doy"] = fd.dayofyear
            row["month"] = fd.month
            row["doy_sin"] = np.sin(2 * np.pi * fd.dayofyear / 365.25)
            row["doy_cos"] = np.cos(2 * np.pi * fd.dayofyear / 365.25)
            row["is_monsoon"] = int(fd.month in [6, 7, 8, 9])

            if fd.dayofyear in doy_clim.index:
                row["target_doy_mean"] = float(doy_clim.loc[fd.dayofyear, "mean"])
                row["target_doy_std"] = float(doy_clim.loc[fd.dayofyear, "std"])
            else:
                row["target_doy_mean"] = float(train.mean())
                row["target_doy_std"] = float(train.std())

            X_step = np.array([row.get(c, 0) for c in feat_cols]).reshape(1, -1)
            X_step = np.nan_to_num(X_step, nan=0.0)
            try:
                pred = float(model.predict(X_step)[0])
            except Exception:
                pred = float(history.iloc[-1])

            if target == "precip":
                pred = max(pred, 0)
            elif target in ("flood_event", "drought_event", "heatwave_event", "agri_event"):
                pred = float(np.clip(pred, 0, 1))

            days_out = (fd - last_date).days
            unc = rmse_hold * (1 + days_out / 30)
            forecasts.append({
                "date": fd,
                "forecast": pred,
                "lower": pred - 1.96 * unc,
                "upper": pred + 1.96 * unc,
            })

            history = pd.concat([history, pd.Series([pred], index=[fd])])

        return pd.DataFrame(forecasts)
