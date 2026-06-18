# models_xgb.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError("xgboost ist nicht installiert. Bitte `pip install xgboost` ausführen.") from e

from sklearn.multioutput import MultiOutputRegressor
from config import EVAL_HORIZONS_DEFAULT, XGB_PARAMS

__all__ = ["DirectForecasterXGB", "MIMOForecasterXGB"]


# --- XGBoost-Param-Sanitizer (Version-kompatibel) ----------------------
def _sanitize_xgb_params(p: dict) -> dict:
    """Macht Params kompatibel zu XGBoost <2.0 und >=2.0."""
    q = dict(p or {})
    try:
        import xgboost as xgb
        from packaging import version as V
        v = V.parse(xgb.__version__)
    except Exception:
        return q  # zur Not unverändert

    if v >= V.parse("2.0.0"):
        # In v2 gilt: 'device' benutzen (z.B. 'cuda'); KEIN gpu_id/predictor nötig
        if "device" in q:
            q.pop("gpu_id", None)
            q.pop("predictor", None)
            # 'gpu_hist' explizit ist nicht nötig → auf 'hist' zurücksetzen
            if q.get("tree_method") == "gpu_hist":
                q["tree_method"] = "hist"
    else:
        # Legacy: KEIN 'device', stattdessen tree_method/predictor/gpu_id
        q.pop("device", None)
    return q


# ---------------------------------------------------------------------
# Direct: ein separates XGBRegressor pro Horizont
# ---------------------------------------------------------------------
@dataclass
class DirectForecasterXGB:
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT
    model_params: Optional[Dict] = None
    feature_cols_: Optional[List[str]] = None
    models_: Dict[int, XGBRegressor] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # defensive copy + Grundnormalisierung
        self.model_params = dict(self.model_params or XGB_PARAMS)
        self._normalize_params()
        self.horizons = tuple(int(h) for h in self.horizons)

    def _normalize_params(self) -> None:
        p = self.model_params
        # sichere Defaults
        p.setdefault("objective", "reg:squarederror")
        p.setdefault("n_jobs", XGB_PARAMS.get("n_jobs", 6))
        p.setdefault("verbosity", XGB_PARAMS.get("verbosity", 0))  # 0 = silent
        # hist/gpu_hist je nach device
        dev = p.get("device", XGB_PARAMS.get("device"))
        if dev == "cuda":
            p.setdefault("tree_method", "gpu_hist")
            p.setdefault("predictor", "gpu_predictor")
        else:
            p.setdefault("tree_method", XGB_PARAMS.get("tree_method", "hist"))

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "DirectForecasterXGB":
        if not feature_cols:
            raise ValueError("feature_cols darf nicht leer sein.")
        self.feature_cols_ = list(feature_cols)

        X = df_train.loc[:, self.feature_cols_].to_numpy(copy=False)
        y_cols = [f"y+{h}" for h in self.horizons]
        Y = df_train.loc[:, y_cols].to_numpy(copy=False)  # shape (n, H)

        self.models_.clear()
        for i, h in enumerate(self.horizons):
            m = XGBRegressor(**_sanitize_xgb_params(self.model_params))
            m.fit(X, Y[:, i])
            self.models_[h] = m
        return self

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None or not self.models_:
            raise RuntimeError("predict() vor fit() aufgerufen oder keine Modelle vorhanden.")
        X = df_in.loc[:, self.feature_cols_].to_numpy(copy=False)

        n = X.shape[0]
        k = len(self.horizons)
        Yhat = np.empty((n, k), dtype=float)

        for j, h in enumerate(self.horizons):
            Yhat[:, j] = self.models_[h].predict(X)

        cols = [f"y+{h}" for h in self.horizons]
        return pd.DataFrame(Yhat, index=df_in.index, columns=cols)


# ---------------------------------------------------------------------
# MIMO: ein Modell mit MultiOutputRegressor(XGBRegressor)
# ---------------------------------------------------------------------
@dataclass
class MIMOForecasterXGB:
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT
    model_params: Optional[Dict] = None
    mor_n_jobs: int = 1
    feature_cols_: Optional[List[str]] = None
    model_: Optional[MultiOutputRegressor] = None

    def __post_init__(self):
        # defensive copy + Grundnormalisierung
        self.model_params = dict(self.model_params or XGB_PARAMS)
        self._normalize_params()
        self.horizons = tuple(int(h) for h in self.horizons)

    def _normalize_params(self) -> None:
        p = self.model_params
        # sichere Defaults
        p.setdefault("objective", "reg:squarederror")
        p.setdefault("n_jobs", XGB_PARAMS.get("n_jobs", 6))
        p.setdefault("verbosity", XGB_PARAMS.get("verbosity", 0))  # 0 = silent
        dev = p.get("device", XGB_PARAMS.get("device"))
        if dev == "cuda":
            p.setdefault("tree_method", "gpu_hist")
            p.setdefault("predictor", "gpu_predictor")
        else:
            p.setdefault("tree_method", XGB_PARAMS.get("tree_method", "hist"))

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "MIMOForecasterXGB":
        if not feature_cols:
            raise ValueError("feature_cols darf nicht leer sein.")
        self.feature_cols_ = list(feature_cols)

        X = df_train.loc[:, self.feature_cols_].to_numpy(copy=False)
        y_cols = [f"y+{h}" for h in self.horizons]
        Y = df_train.loc[:, y_cols].to_numpy(copy=False)

        base = XGBRegressor(**_sanitize_xgb_params(self.model_params))
        self.model_ = MultiOutputRegressor(base, n_jobs=self.mor_n_jobs)
        self.model_.fit(X, Y)
        return self

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None or self.model_ is None:
            raise RuntimeError("predict() vor fit() aufgerufen oder Modell fehlt.")
        X = df_in.loc[:, self.feature_cols_].to_numpy(copy=False)
        Yhat = self.model_.predict(X)
        cols = [f"y+{h}" for h in self.horizons]
        return pd.DataFrame(Yhat, index=df_in.index, columns=cols)

# Ende models_xgb.py
