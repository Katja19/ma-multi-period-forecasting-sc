# models_dl_darts.py
from __future__ import annotations

"""
Darts-basierte DL-Wrapper:
- _BaseDartsGlobal: minimaler gemeinsamer Nenner für *ein* Kovariatetyp (past ODER future)
- NHiTSForecaster: nutzt past_covariates → erbt Baselogik
- TFTForecaster  : benötigt past UND future → eigene fit/predict-Implementierung

Design-Ziel: Basisklasse schlank halten; TFT bleibt größer (andere Kovariaten-Mechanik).
"""

# ============================== Imports ==============================
# Stdlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Third-party
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandas.tseries.frequencies import to_offset
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

import re

# Darts / Torch (hart benötigt für diese Datei)
try:
    from darts import TimeSeries
    from darts.models import NHiTSModel, TFTModel
    import torch
except Exception as e:
    raise ImportError(
        'Darts/Torch fehlt. Bitte installieren: pip install "u8darts[torch]" torch pytorch-lightning'
    ) from e

# Project
from datasets import split_features_for_dl, detect_dataset_from_cols
from config import (
    DATE_COL, TARGET_COL, SERIES_COLS,
    EVAL_HORIZONS_DEFAULT, RANDOM_STATE, AUX_KEY_COLS, save_dl_val_summary
)

def _split_tft_from_config(all_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Teilt Feature-Spalten in (future_known, past_only) gemäß FUTURE_TFT/PAST_TFT
    aus config.py. Die Einträge in FUTURE_TFT/PAST_TFT sind Regex-Patterns.
    Bei Überschneidung gewinnt 'future'.
    """
    cols = list(map(str, all_cols or []))

    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        return any(re.search(p, name) for p in patterns)

    future = [c for c in cols if _matches_any(c, FUTURE_TFT)]
    future_set = set(future)
    past   = [c for c in cols if (_matches_any(c, PAST_TFT) and c not in future_set)]
    return future, past

__all__ = ["NHiTSForecaster", "TFTForecaster"]

# ============================== Helpers ==============================

_TREE_PARAM_KEYS = {
    # Tree/HW/Projekt-Flags, die NIE in DL-Modelle dürfen:
    "n_jobs","nthread","n_estimators","num_boost_round","learning_rate","eta",
    "max_depth","num_leaves","max_bin","subsample","colsample_bytree",
    "colsample_bylevel","colsample_bynode","reg_alpha","reg_lambda","lambda_l1","lambda_l2",
    "min_child_weight","min_child_samples","verbosity","verbose",
    "tree_method","predictor","objective","boosting_type",
    "device","device_type","gpu_platform_id","gpu_device_id",
    "bagging_freq","single_precision_histogram",
    "FORCE_TREES_ON_GPU","PREFERRED_GPU_ID",
    # nur beim FIT erlaubt, nicht im __init__:
}

def _clean_darts_params(p: dict) -> dict:
    if not isinstance(p, dict):
        return {}
    q = {k: v for k, v in p.items() if k not in _TREE_PARAM_KEYS}
    allowed = {
        "input_chunk_length","output_chunk_length","output_chunk_shift",
        "hidden_size","lstm_layers","num_attention_heads",
        "dropout","activation","n_epochs","batch_size",
        "optimizer_kwargs","loss_fn","random_state","pl_trainer_kwargs",
        # N-HiTS:
        "num_stacks","num_blocks","num_layers","layer_widths",
        "pooling_kernel_sizes","n_freq_downsample","MaxPool1d",
        "use_static_covariates",
    }
    return {k: v for k, v in q.items() if k in allowed}


def _as_series_lists(df: pd.DataFrame, feature_cols: List[str]):
    """Baut pro Serie eine TimeSeries für y und (optional) eine Kovariat-TS aus feature_cols."""
    df = df.sort_values(list(SERIES_COLS) + [DATE_COL])
    ids, y_list = [], []
    cov_list: Optional[List[TimeSeries]] = [] if feature_cols else None

    for gid, g in df.groupby(list(SERIES_COLS), sort=False, observed=False):
        uid = gid if isinstance(gid, str) else "_".join(map(str, np.atleast_1d(gid)))
        ids.append(uid)

        # y
        ts_y = TimeSeries.from_times_and_values(
            times=DatetimeIndex(pd.to_datetime(g[DATE_COL])),
            values=g[TARGET_COL].to_numpy(dtype=np.float32, copy=False),
            columns=[TARGET_COL],
            fill_missing_dates=True,
            freq="D",
        ).astype(np.float32)
        y_list.append(ts_y)

        # Kovariaten
        if feature_cols:
            X = g.loc[:, feature_cols].copy()
            for c in X.columns:
                if str(X[c].dtype) == "category":
                    X[c] = X[c].cat.codes
            X = X.astype(np.float32)
            ts_c = TimeSeries.from_times_and_values(
                times=DatetimeIndex(pd.to_datetime(g[DATE_COL])),
                values=X.to_numpy(dtype=np.float32, copy=False),
                columns=list(map(str, feature_cols)),
                fill_missing_dates=True,
                freq="D",
            ).astype(np.float32)
            cov_list.append(ts_c)

        # optionale statische Kovariaten an y / cov anhängen (falls vorhanden)
        static = {}
        for k in AUX_KEY_COLS:
            if k not in g.columns:
                continue
            s = g[k]
            v = s.iloc[0]
            # numerisch → float
            if pd.api.types.is_numeric_dtype(s.dtype):
                try:
                    static[k] = [float(v)]
                except Exception:
                    static[k] = [0.0]
            # kategorisch → konsistente Kategorie-Codes (global über df definiert)
            elif str(s.dtype) == "category":
                code = int(s.cat.codes.iloc[0])
                static[k] = [code]
            # sonst (object/str) → stabiler Hash → int
            else:
                code = pd.util.hash_pandas_object(pd.Series([str(v)]), index=False)\
                            .astype("int64").iloc[0] & 0x7FFFFFFF
                static[k] = [int(code)]

        if static:
            static_df = pd.DataFrame(static)
            y_list[-1] = y_list[-1].with_static_covariates(static_df)
            if feature_cols:
                cov_list[-1] = cov_list[-1].with_static_covariates(static_df)

    return ids, y_list, cov_list

def _trainer_kwargs():
    print("Using Torch Trainer (GPU if available), tqdm progress bar.")
    kw = dict(enable_progress_bar=True, precision="32-true",
              log_every_n_steps=5, enable_checkpointing=True)
    try:
        import torch
        if torch.cuda.is_available():
            kw.update(dict(accelerator="gpu", devices=1))
        else:
            kw.update(dict(accelerator="cpu"))
    except Exception:
        kw.update(dict(accelerator="cpu"))

    try:
        kw["callbacks"] = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
        ]
    except Exception:
        pass
    return kw

def _cov_from_cols_for_group(g: pd.DataFrame, cols: Optional[List[str]]) -> Optional[TimeSeries]:
    """Hilfsfunktion: baut eine TimeSeries-Kovariat für *eine* Gruppen-Partition."""
    if not cols:
        return None
    X = g.loc[:, cols].copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes
    X = X.astype(np.float32)
    ts = TimeSeries.from_times_and_values(
        times=DatetimeIndex(pd.to_datetime(g[DATE_COL])),
        values=X.to_numpy(dtype=np.float32, copy=False),
        columns=list(map(str, cols)),
        fill_missing_dates=True,
        freq="D",
    ).astype(np.float32)
    return ts


# ============================== Basisklasse ==============================
@dataclass
class _BaseDartsGlobal:
    """
    Minimal gemeinsame Logik für Darts-Modelle mit *einem* Kovariatetyp (past ODER future).
    NHiTS nutzt diese Klasse direkt; TFT überschreibt fit/predict aufgrund dualer Kovariaten.
    """
    def __init__(self,
                 horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT,
                 model_params: Optional[Dict] = None,
                 **_):
        self.horizons = tuple(int(h) for h in horizons)
        self.H = int(max(self.horizons))
        self.model_params = dict(model_params or {})
        self.feature_cols_: Optional[List[str]] = None
        self.model_ = None

        # Welche Art von Kovariaten dieser Wrapper nutzt (Default 'future', NHiTS setzt 'past')
        self.covariate_kind = getattr(self, "covariate_kind", "future")

        # Train-Speicher
        self._train_ids: List[str] = []
        self._train_y: List[TimeSeries] = []
        self._train_cov: Optional[List[TimeSeries]] = None  # ein Typ (past ODER future)
        self._covariate_cols_: Optional[List[str]] = None   # exakt die im Training genutzten Spalten

    def _build_model(self, **p):
        raise NotImplementedError

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "_BaseDartsGlobal":
        print(f"Fitting Darts model {self.__class__.__name__} for horizons {self.horizons} ...")
        self.feature_cols_ = list(feature_cols or [])

        # Dataset-Heuristik für Feature-Split
        dataset_name = detect_dataset_from_cols(df_train)
        future_known, past_only = split_features_for_dl(self.feature_cols_, dataset_name)  # mode="generic"

        # Für Basisklasse: genau EIN Kovariaten-Typ verwenden
        used_cov_cols = future_known if self.covariate_kind == "future" else past_only
        self._covariate_cols_ = list(used_cov_cols)

        # Serien & (optionale) Kovariaten bauen
        ids, ys, covs = _as_series_lists(df_train, used_cov_cols)
        self._train_ids, self._train_y = ids, ys
        self._train_cov = covs if (covs and len(covs) > 0) else None

        # Debug
        print(f"[DL] covariate_kind={self.covariate_kind}; used_cov_cols={len(used_cov_cols)}")
        print(f"[DL] series count: {len(self._train_y)}")
        print(f"[DL] first target length: {self._train_y[0].n_timesteps}")
        if self._train_cov is not None:
            print(f"[DL] first cov shape: {self._train_cov[0].n_timesteps} x {self._train_cov[0].n_components}")
        else:
            print("[DL] no covariates used")

        # Modell erstellen
        p = dict(self.model_params)
        p.setdefault("output_chunk_length", self.H)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("pl_trainer_kwargs", _trainer_kwargs())
        self.model_ = self._build_model(**_clean_darts_params(p))

        # Fit
        fit_kwargs = {}
        if self._train_cov is not None:
            key = "future_covariates" if self.covariate_kind == "future" else "past_covariates"
            fit_kwargs[key] = self._train_cov

        self.model_.fit(series=self._train_y, **fit_kwargs)
        
        return self

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("predict() vor fit().")

        df_in = df_in.sort_values(list(SERIES_COLS) + [DATE_COL]).copy()
        ids_in = (
            df_in[list(SERIES_COLS)].astype(str).agg("_".join, axis=1)
            if len(SERIES_COLS) > 1 else df_in[SERIES_COLS[0]].astype(str)
        )
        start_dates = df_in.groupby(ids_in, sort=False)[DATE_COL].min().map(pd.to_datetime)

        # Map: uid -> (train_y, train_cov)
        train_map = {
            uid: (self._train_y[i], (self._train_cov[i] if self._train_cov else None))
            for i, uid in enumerate(self._train_ids)
        }

        records = []
        for uid, g in df_in.groupby(ids_in, sort=False):
            if uid not in train_map:
                continue
            train_y, train_cov = train_map[uid]

            # Ziel verlängern
            y_ext = TimeSeries.from_times_and_values(
                times=DatetimeIndex(pd.to_datetime(g[DATE_COL])),
                values=g[TARGET_COL].to_numpy(dtype=np.float32, copy=False),
                columns=[TARGET_COL],
                fill_missing_dates=True,
                freq="D",
            ).astype(np.float32)
            series_full = train_y.append(y_ext)

            # Kovariaten nur, wenn im Training vorhanden
            cov_full = None
            if (self._train_cov is not None) and self._covariate_cols_:
                ext = _cov_from_cols_for_group(g, self._covariate_cols_)
                cov_full = train_cov.append(ext) if (train_cov is not None and ext is not None) else (train_cov or ext)

            hist_kwargs = {}
            if cov_full is not None:
                key = "future_covariates" if self.covariate_kind == "future" else "past_covariates"
                hist_kwargs[key] = cov_full

            fcs_list = self.model_.historical_forecasts(
                series=series_full,
                start=start_dates.loc[uid],
                forecast_horizon=self.H,
                stride=1,
                retrain=False,
                verbose=False,
                last_points_only=False,
                overlap_end=True,
                **hist_kwargs,
            )

            off = to_offset(series_full.freq_str or "D")
            series_key_vals = g.iloc[0][list(SERIES_COLS)].to_dict() if len(SERIES_COLS) > 1 else None

            for ts in fcs_list:
                origin_dt = ts.time_index[0] - off
                vals = ts.values(copy=False)
                if vals.ndim == 3:
                    vals = vals[:, :, 0]
                vals = vals.astype(np.float32, copy=False).ravel()

                row = {DATE_COL: origin_dt}
                if len(SERIES_COLS) == 1:
                    row[SERIES_COLS[0]] = g.iloc[0][SERIES_COLS[0]]
                else:
                    row.update(series_key_vals)

                for j in range(self.H):
                    row[f"y+{j+1}"] = float(vals[j]) if j < len(vals) else np.nan
                records.append(row)

        Yhat = pd.DataFrame.from_records(records)
        out = (
            df_in.loc[:, list(SERIES_COLS) + [DATE_COL]]
            .merge(Yhat, on=list(SERIES_COLS) + [DATE_COL], how="left")
            .set_index(df_in.index)
        )
        return out[[f"y+{h}" for h in self.horizons]]


# ============================== Konkrete Modelle ==============================

# --- N-HiTS: FAST Preset ---
class NHiTSForecaster(_BaseDartsGlobal):
    covariate_kind = "past"
    def _build_model(self, **p):
        p.setdefault("input_chunk_length", 28)  # 28 -> 21 (weniger History)
        p.setdefault("output_chunk_length", self.H)
        p.setdefault("n_epochs", 25)      # 25 -> 10  (≈ 2x schneller)
        p.setdefault("batch_size", 128)   # 64 -> 128 (weniger Steps)
        p.setdefault("dropout", 0.10)
        p.setdefault("optimizer_kwargs", {"lr": 1e-3})
        print("[NHiTS] n_epochs=", p["n_epochs"], ", batch_size=", p["batch_size"])
        return NHiTSModel(**_clean_darts_params(p))

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "_BaseDartsGlobal":
        """Wie Basisklasse, aber mit expliziter Validation (val_series & val_past_covariates)."""
        print(f"Fitting Darts model {self.__class__.__name__} for horizons {self.horizons} ...")
        self.feature_cols_ = list(feature_cols or [])

        # 1) Feature-Split (wir nutzen nur 'past' für N-HiTS)
        dataset_name = detect_dataset_from_cols(df_train)
        future_known, past_only = split_features_for_dl(self.feature_cols_, dataset_name)  # mode="generic"
        used_cov_cols = past_only
        self._covariate_cols_ = list(used_cov_cols)

        # 2) Serien & (optionale) Kovariaten (volle Länge für spätere predict-Phase behalten)
        ids, ys, covs = _as_series_lists(df_train, used_cov_cols)
        self._train_ids, self._train_y = ids, ys
        self._train_cov = covs if (covs and len(covs) > 0) else None

        # 3) Modell bauen (Trainer mit EarlyStopping/Checkpoint ist in _trainer_kwargs)
        p = dict(self.model_params)
        p.setdefault("output_chunk_length", self.H)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("pl_trainer_kwargs", _trainer_kwargs())

        # 4) Zeitbasierter Val-Split (einfach & robust)
        self.model_ = self._build_model(**_clean_darts_params(p))
        val_days = max(28, self.H)
        ys_tr  = [s[:-val_days] for s in self._train_y]
        ys_val = [s for s in self._train_y]
        past_tr = [c[:-val_days] for c in self._train_cov] if self._train_cov else None
        past_val = self._train_cov if self._train_cov else None
        
        # 5) Fit mit Validation (aktiviert EarlyStopping/Best-Checkpoint)

        self.model_.fit(
            series=ys_tr,
            past_covariates=past_tr,
            val_series=ys_val,
            val_past_covariates=past_val,
        )
        
        # 6) NEU: DL-Validation-Summary sichern (kein HPO)
        # Dataset-Name wurde oben bereits ermittelt: dataset_name = detect_dataset_from_cols(df_train)
        # Hyperparam-Whitelist (primitive Felder → JSON-sicher)
        used_hp = {}
        for k in ("input_chunk_length", "output_chunk_length", "n_epochs", "batch_size", "dropout",
                  "optimizer_kwargs", "random_state"):
            if k in p:
                used_hp[k] = p[k]

        save_dl_val_summary(
            dataset_name=dataset_name,
            model_name="nhits",
            wrapper="native",
            used_params=used_hp,
            val_days=val_days,
            train_meta={"early_stopping": True, "checkpointing": True}
        )
        
        return self
#####################################################################################

# --- TFT: FAST Preset ---
class TFTForecaster(_BaseDartsGlobal):
    def _build_model(self, **p):
        p.setdefault("input_chunk_length", 84) # größer schneller
        p.setdefault("output_chunk_length", self.H)
        p.setdefault("hidden_size", 8)       # kleiner schneller
        p.setdefault("lstm_layers", 1)
        p.setdefault("num_attention_heads", 1) # 2 -> 1  (schneller)
        p.setdefault("n_epochs", 6)          # 30 -> 10 (≈ 2x schneller)
        p.setdefault("batch_size", 256)       # 128 -> 256 (größer schneller)
        p.setdefault("dropout", 0.10)
        p.setdefault("optimizer_kwargs", {"lr": 2e-3, "weight_decay": 1e-4})
        p.setdefault("loss_fn", torch.nn.SmoothL1Loss())
        p.setdefault("use_static_covariates", False)  # statische Kovariaten ignorieren # neu
        print("[TFT] n_epochs=", p["n_epochs"], ", batch_size=", p["batch_size"])
        return TFTModel(**_clean_darts_params(p))

    # Container
    _train_past_cov: Optional[List[TimeSeries]] = None
    _train_future_cov: Optional[List[TimeSeries]] = None
    _past_cols_: Optional[List[str]] = None
    _future_cols_: Optional[List[str]] = None

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "_BaseDartsGlobal":
        print(f"Fitting Darts model {self.__class__.__name__} for horizons {self.horizons} ...")
        self.feature_cols_ = list(feature_cols or [])
        
        # 1) Feature-Split (past/future) anhand Dataset-Heuristik
        dataset_name = detect_dataset_from_cols(df_train)
        future_known, past_only = split_features_for_dl(self.feature_cols_, dataset_name, mode="tft")
        # FUTURE_TFT / PAST_TFT aus config.py verwenden (keine Heuristik)
        #future_known, past_only = _split_tft_from_config(self.feature_cols_)
        
        print(f"[TFT] Using {len(future_known)} future-known and {len(past_only)} past-only covariates.")

        self._past_cols_, self._future_cols_ = list(past_only), list(future_known)

        # 2a) Targets unabhängig von Kovariaten
        ids, ys, _ = _as_series_lists(df_train, [])
        self._train_ids, self._train_y = ids, ys

        # 2b) Kovariaten-Listen aufbauen
        def _build_cov_list(df: pd.DataFrame, cols: List[str]) -> Optional[List[TimeSeries]]:
            if not cols:
                return None
            df = df.sort_values(list(SERIES_COLS) + [DATE_COL])
            covs: List[TimeSeries] = []
            for _, g in df.groupby(list(SERIES_COLS), sort=False, observed=False):
                ts = _cov_from_cols_for_group(g, cols)
                covs.append(ts)
            return covs

        self._train_past_cov   = _build_cov_list(df_train, self._past_cols_)
        self._train_future_cov = _build_cov_list(df_train, self._future_cols_)

        # 3) Modell bauen (vor dem ersten fit!)
        p = dict(self.model_params)
        p.setdefault("output_chunk_length", self.H)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("pl_trainer_kwargs", _trainer_kwargs())

        # 4) Einfacher Val-Split (zeitbasiert)
        self.model_ = self._build_model(**_clean_darts_params(p))
        val_days = max(28, self.H)
        ys_tr, ys_val = [s[:-val_days] for s in self._train_y], [s for s in self._train_y]
        def _split(covs): return ([c[:-val_days] for c in covs], covs) if covs else (None, None)
        past_tr, past_val = _split(self._train_past_cov)
        fut_tr,  fut_val  = _split(self._train_future_cov)

        # 5) Train mit Validation
        self.model_.fit(
            series=ys_tr,
            past_covariates=past_tr,
            future_covariates=fut_tr,
            val_series=ys_val,
            val_past_covariates=past_val,
            val_future_covariates=fut_val,
        )

        # 6) NEU: DL-Validation-Summary sichern
        used_hp = {}
        for k in ("input_chunk_length", "output_chunk_length", "hidden_size", "lstm_layers",
                  "num_attention_heads", "n_epochs", "batch_size", "dropout",
                  "optimizer_kwargs", "random_state"):
            if k in p:
                used_hp[k] = p[k]

        save_dl_val_summary(
            dataset_name=dataset_name,
            model_name="tft",
            wrapper="native",
            used_params=used_hp,
            val_days=val_days,
            train_meta={"early_stopping": True, "checkpointing": True}
        )
        
        return self

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("predict() vor fit().")

        df_in = df_in.sort_values(list(SERIES_COLS) + [DATE_COL]).copy()
        ids_in = (
            df_in[list(SERIES_COLS)].astype(str).agg("_".join, axis=1)
            if len(SERIES_COLS) > 1 else df_in[SERIES_COLS[0]].astype(str)
        )
        start_dates = df_in.groupby(ids_in, sort=False)[DATE_COL].min().map(pd.to_datetime)

        # Map: uid -> (y, past, future)
        train_map = {
            uid: (
                self._train_y[i],
                self._train_past_cov[i] if self._train_past_cov else None,
                self._train_future_cov[i] if self._train_future_cov else None,
            )
            for i, uid in enumerate(self._train_ids)
        }

        records = []
        for uid, g in df_in.groupby(ids_in, sort=False):
            if uid not in train_map:
                continue
            train_y, train_past, train_future = train_map[uid]

            # Ziel verlängern
            y_ext = TimeSeries.from_times_and_values(
                times=DatetimeIndex(pd.to_datetime(g[DATE_COL])),
                values=g[TARGET_COL].to_numpy(dtype=np.float32, copy=False),
                columns=[TARGET_COL],
                fill_missing_dates=True,
                freq="D",
            ).astype(np.float32)
            series_full = train_y.append(y_ext)

            # Test-Kovariaten
            past_ext   = _cov_from_cols_for_group(g, self._past_cols_)
            future_ext = _cov_from_cols_for_group(g, self._future_cols_)

            past_full   = train_past.append(past_ext)     if (train_past is not None and past_ext is not None) else (train_past or past_ext)
            future_full = train_future.append(future_ext) if (train_future is not None and future_ext is not None) else (train_future or future_ext)

            hist_kwargs = {}
            if past_full   is not None: hist_kwargs["past_covariates"]   = past_full
            if future_full is not None: hist_kwargs["future_covariates"] = future_full

            fcs_list = self.model_.historical_forecasts(
                series=series_full,
                start=start_dates.loc[uid],
                forecast_horizon=self.H,
                stride=1,
                retrain=False,
                verbose=False,
                last_points_only=False,
                overlap_end=True,
                **hist_kwargs,
            )

            off = to_offset(series_full.freq_str or "D")
            series_key_vals = g.iloc[0][list(SERIES_COLS)].to_dict() if len(SERIES_COLS) > 1 else None

            for ts in fcs_list:
                origin_dt = ts.time_index[0] - off
                vals = ts.values(copy=False)
                if vals.ndim == 3:
                    vals = vals[:, :, 0]
                vals = vals.astype(np.float32, copy=False).ravel()

                row = {DATE_COL: origin_dt}
                if len(SERIES_COLS) == 1:
                    row[SERIES_COLS[0]] = g.iloc[0][SERIES_COLS[0]]
                else:
                    row.update(series_key_vals)

                for j in range(self.H):
                    row[f"y+{j+1}"] = float(vals[j]) if j < len(vals) else np.nan
                records.append(row)

        Yhat = pd.DataFrame.from_records(records)
        out = (
            df_in.loc[:, list(SERIES_COLS) + [DATE_COL]]
            .merge(Yhat, on=list(SERIES_COLS) + [DATE_COL], how="left")
            .set_index(df_in.index)
        )
        return out[[f"y+{h}" for h in self.horizons]]

# Ende models_dl_darts.py
