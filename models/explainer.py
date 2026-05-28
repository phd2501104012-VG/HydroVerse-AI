from typing import Optional, Dict, List, Any, Union
import numpy as np
import pandas as pd

from utils.logger import log


class ModelExplainer:
    def __init__(self):
        self._has_shap = False
        self._check_deps()

    def _check_deps(self):
        try:
            import shap
            self._has_shap = True
        except ImportError:
            pass

    def explain_with_shap(self, model, X: np.ndarray, feature_names: List[str]) -> Optional[Dict]:
        if not self._has_shap:
            log.warning("SHAP not available")
            return None

        try:
            import shap
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)

            shap_values = explainer(X[:100] if len(X) > 100 else X)

            return {
                "shap_values": shap_values.values,
                "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0,
                "feature_importance": dict(zip(
                    feature_names,
                    np.abs(shap_values.values).mean(axis=0) if len(shap_values.values.shape) > 1 else np.abs(shap_values.values),
                )),
            }
        except Exception as e:
            log.warning(f"SHAP explanation failed: {e}")
            return None

    def feature_importance(self, model, feature_names: List[str]) -> Optional[pd.DataFrame]:
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten() if model.coef_ is not None else None

        if importances is None or len(importances) != len(feature_names):
            return None

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def permutation_importance(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        metric: str = "rmse",
    ) -> pd.DataFrame:
        from sklearn.metrics import mean_squared_error
        from sklearn.inspection import permutation_importance

        try:
            result = permutation_importance(
                model, X_val, y_val,
                n_repeats=n_repeats,
                scoring="neg_mean_squared_error" if metric == "rmse" else "r2",
                random_state=42,
            )
            return pd.DataFrame({
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        except Exception as e:
            log.warning(f"Permutation importance failed: {e}")
            return pd.DataFrame()

    def uncertainty_quantification(
        self,
        ensemble_preds: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        mean = np.mean(ensemble_preds, axis=0)
        std = np.std(ensemble_preds, axis=0)
        z = 1.96  # 95% CI

        return {
            "mean": mean,
            "std": std,
            "lower": mean - z * std,
            "upper": mean + z * std,
            "cv": std / (np.abs(mean) + 1e-10),
        }
