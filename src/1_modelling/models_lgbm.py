# models_lgbm.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from config import EVAL_HORIZONS_DEFAULT, LGBM_PARAMS

__all__ = ["DirectForecaster", "MIMOForecaster"]


# ---------------------------- Helpers ----------------------------

def _strip_lgbm_gpu_params(p: dict) -> dict:
    """
    Entfernt GPU-bezogene LightGBM-Parameter (device_type, gpu_platform_id, gpu_device_id).
    Praktisch, wenn eine CPU-Only-Build verwendet wird, aber GPU-Keys gesetzt sind.
    """
    q = dict(p or {})
    for k in ("device_type", "gpu_platform_id", "gpu_device_id"):
        q.pop(k, None)
    return q


# ---------------------------------------------------------------------
# Direct: ein separates LGBM pro Horizont
# ---------------------------------------------------------------------
@dataclass
class DirectForecaster:
    """
    Trainiert pro Horizont h ein eigenes LGBMRegressor-Modell.

    Optimierungen:
      - X/Y einmal extrahieren (NumPy) → weniger Pandas-Overhead
      - stabile Reihenfolge der Horizonte

    Tipp für schwächere PCs:
      - Belasse 'n_jobs' in LGBM_PARAMS moderat (z. B. 2)
      - keine äußere Parallelisierung
    """
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT
    model_params: Optional[Dict] = None
    feature_cols_: Optional[List[str]] = None
    models_: Dict[int, LGBMRegressor] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # defensive copy der Params (keine In-Place-Mutation)
        self.model_params = dict(self.model_params or LGBM_PARAMS)
        # Horizonte stabilisieren
        self.horizons = tuple(int(h) for h in self.horizons)

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "DirectForecaster":
        if not feature_cols:
            raise ValueError("feature_cols darf nicht leer sein.")
        self.feature_cols_ = list(feature_cols)

        # Einmalig extrahieren (schneller als wiederholtes Spalten-Slicing)
        X = df_train.loc[:, self.feature_cols_].to_numpy(copy=False)
        y_cols = [f"y+{h}" for h in self.horizons]
        Y = df_train.loc[:, y_cols].to_numpy(copy=False)  # shape (n, H)

        self.models_.clear()
        for i, h in enumerate(self.horizons):
            # Erst mit evtl. GPU-Params probieren …
            params_try = dict(self.model_params)
            try:
                m = LGBMRegressor(**params_try)
                m.fit(X, Y[:, i])
            except Exception as e:
                msg = str(e)
                # … wenn die Build keine GPU kann: Fallback ohne GPU-Params
                if ("GPU" in msg) or ("gpu" in msg) or ("Device" in msg):
                    params_cpu = _strip_lgbm_gpu_params(params_try)
                    m = LGBMRegressor(**params_cpu)
                    m.fit(X, Y[:, i])
                else:
                    raise
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
# MIMO: ein Modell mit MultiOutputRegressor
# ---------------------------------------------------------------------
@dataclass
class MIMOForecaster:
    """
    Nutzt MultiOutputRegressor(LightGBM) für mehrere Ziele gleichzeitig.

    Optimierungen:
      - X/Y einmalig als NumPy
      - stabile Reihenfolge der Horizonte

    Tipp:
      - mor_n_jobs=1 belassen und LightGBM-Threads in LGBM_PARAMS dosieren.
    """
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT
    model_params: Optional[Dict] = None
    mor_n_jobs: int = 1  # parallelisiert über Ziele; 1 ist am schonendsten
    feature_cols_: Optional[List[str]] = None
    model_: Optional[MultiOutputRegressor] = None

    def __post_init__(self):
        # defensive copy
        self.model_params = dict(self.model_params or LGBM_PARAMS)
        self.horizons = tuple(int(h) for h in self.horizons)

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "MIMOForecaster":
        if not feature_cols:
            raise ValueError("feature_cols darf nicht leer sein.")
        self.feature_cols_ = list(feature_cols)

        X = df_train.loc[:, self.feature_cols_].to_numpy(copy=False)
        y_cols = [f"y+{h}" for h in self.horizons]
        Y = df_train.loc[:, y_cols].to_numpy(copy=False)

        # Erst mit evtl. GPU-Params probieren …
        params_try = dict(self.model_params)
        base = LGBMRegressor(**params_try)
        momo = MultiOutputRegressor(base, n_jobs=self.mor_n_jobs)
        try:
            momo.fit(X, Y)
            self.model_ = momo
            return self
        except Exception as e:
            msg = str(e)
            # Fallback ohne GPU-Params
            if ("GPU" in msg) or ("gpu" in msg) or ("Device" in msg):
                params_cpu = _strip_lgbm_gpu_params(params_try)
                base2 = LGBMRegressor(**params_cpu)
                momo2 = MultiOutputRegressor(base2, n_jobs=self.mor_n_jobs)
                momo2.fit(X, Y)
                self.model_ = momo2
                return self
            raise

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None or self.model_ is None:
            raise RuntimeError("predict() vor fit() aufgerufen oder Modell fehlt.")
        X = df_in.loc[:, self.feature_cols_].to_numpy(copy=False)
        Yhat = self.model_.predict(X)  # shape (n, H)
        cols = [f"y+{h}" for h in self.horizons]
        return pd.DataFrame(Yhat, index=df_in.index, columns=cols)

# Ende models_lgbm.py