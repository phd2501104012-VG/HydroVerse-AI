from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC

from config import CFG
from utils.logger import log


try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class ClassicalModels:
    MODELS = {
        "regression": {
            "RandomForest": lambda rs: RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=rs),
            "ExtraTrees": lambda rs: ExtraTreesRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=rs),
            "GradientBoosting": lambda rs: GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=rs),
            "Ridge": lambda rs: Ridge(alpha=1.0),
        },
        "classification": {
            "RandomForest": lambda rs: RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=rs),
            "ExtraTrees": lambda rs: ExtraTreesClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=rs),
            "GradientBoosting": lambda rs: GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=rs),
            "LogisticRegression": lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
        },
    }

    if HAS_XGB:
        MODELS["regression"]["XGBoost"] = lambda rs: XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbosity=0, n_jobs=-1)
        MODELS["classification"]["XGBoost"] = lambda rs: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbosity=0, n_jobs=-1)
    if HAS_LGB:
        MODELS["regression"]["LightGBM"] = lambda rs: LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbose=-1, n_jobs=-1)
        MODELS["classification"]["LightGBM"] = lambda rs: LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=rs, verbose=-1, n_jobs=-1)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task: str = "regression",
        model_name: str = "RandomForest",
    ) -> Optional[Any]:
        if task not in self.MODELS or model_name not in self.MODELS[task]:
            log.warning(f"Unknown model: {model_name} for {task}")
            return None

        try:
            model = self.MODELS[task][model_name](CFG.ml.random_state)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            log.warning(f"{model_name} training failed: {e}")
            return None

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task: str = "regression",
    ) -> Dict[str, Any]:
        results = {}
        for name in self.MODELS.get(task, {}):
            model = self.train_model(X_train, y_train, task, name)
            if model is not None:
                results[name] = model
        return results
