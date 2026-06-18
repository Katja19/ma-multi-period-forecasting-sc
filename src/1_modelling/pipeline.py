# pipeline.py
from __future__ import annotations
"""
Forecasting-Pipeline (strukturierte Version)
- Saubere Import-Gruppierung und optionale Abhängigkeiten
- Einheitliche Kommentare & klarer Kontrollfluss
- Funktionsgleich zu deiner bisherigen Implementierung
"""

# =============================================================================
# Imports
# =============================================================================
# --- Standardbibliothek
import inspect
import math
import os
import time
from typing import Dict, Sequence

# --- Third-Party
import numpy as np
import pandas as pd

# Optionale Backends
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    xgb = None
    XGBRegressor = None  # type: ignore[assignment]
    HAVE_XGB = False

try:
    import lightgbm as lgbm
    from lightgbm import LGBMRegressor, early_stopping
    HAVE_LGBM = True
except Exception:
    lgbm = None
    LGBMRegressor = None  # type: ignore[assignment]
    early_stopping = None  # type: ignore[assignment]
    HAVE_LGBM = False

# Optionale Systeminfos
try:
    import psutil  # optional
except Exception:
    psutil = None

try:
    import resource  # optional (nur Unix)
except Exception:
    resource = None

# --- Projekt-Lokale Module
from datasets import get_spec
from core import (
    sort_df,
    cast_categoricals,
    make_horizon_targets,
    time_based_origin_split,
    rmsse_scales, rmsse_per_h_and_total,
    count_series,
)
from wandb_utils import (
    init_run,
    log_image, log_metrics, log_feature_importance, log_regression_metrics, log_test_sample,
    log_shap_summary, log_train_profile, log_rmsse_per_series,
    finish_run,
)
from model_utils import (
    feature_importance_df,
    regression_metrics_table,
    regression_metrics_total,
    shap_summary_df,
    shap_values_aggregate,
    shap_beeswarm_figure,
    rmsse_per_series_df,
    detect_compute_backend,
)
from hpo import run_optuna_hpo
from models_xgb import _sanitize_xgb_params

from config import (
    DATE_COL, TARGET_COL, SERIES_COLS,
    EVAL_HORIZONS_DEFAULT, N_TEST_ORIGINS_DEFAULT,
    SEASONAL_PERIOD, RANDOM_STATE,
    USE_HPTUNED_PARAMS,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY, wandb_run_context,
    get_params_for_dataset, set_global_seed,
    AUX_KEY_COLS, VERBOSE, LOG_SHAP, HPO_CFG,
)

# Param-Defaults (weiche Abhängigkeit)
try:
    from config import XGB_PARAMS, LGBM_PARAMS, NHITS_PARAMS, TFT_PARAMS, NAIVE_PARAMS
except Exception:
    XGB_PARAMS = {}
    LGBM_PARAMS = {}
    NHITS_PARAMS = {}
    TFT_PARAMS = {}
    NAIVE_PARAMS = {}


# =============================================================================
# Run-Beobachter Anzeige (ETA, Device, Work Units) Helfer
# =============================================================================
def _gpu_name_str() -> str:
    """Liest (wenn möglich) den GPU-Namen – safe, optional."""
    try:
        import torch
        if torch.cuda.is_available():
            idx = int(torch.cuda.current_device())
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
    except Exception:
        pass
    return "cpu"

def _tree_device_label(resolved_backend: str, params: dict) -> str:
    """
    Liefert ein präzises Device/Method-Label für XGBoost/LightGBM anhand der PARAMS.
    Beispiele:
      - XGBoost >=2.0:   "cuda#0"
      - XGBoost <2.0:    "gpu_hist#0" oder "gpu_predictor#0"
      - LightGBM (GPU):  "gpu#0"
      - CPU (beide):     "cpu"
    """
    rb = (resolved_backend or "").lower()
    p = params or {}

    if rb in ("xgboost", "xgb"):
        dev  = str(p.get("device", "")).lower()          # XGB >= 2.0
        tm   = str(p.get("tree_method", "")).lower()     # XGB < 2.0
        pred = str(p.get("predictor", "")).lower()
        gid  = p.get("gpu_id")

        if dev == "cuda":
            return f"cuda#{gid}" if gid is not None else "cuda"
        if "gpu_hist" in tm:
            return f"gpu_hist#{gid}" if gid is not None else "gpu_hist"
        if "gpu_predictor" in pred:
            return f"gpu_predictor#{gid}" if gid is not None else "gpu_predictor"
        return "cpu"

    if rb in ("lightgbm", "lgbm"):
        dtype = str(p.get("device_type", "")).lower()
        if dtype == "gpu":
            gid = p.get("gpu_device_id")
            return f"gpu#{gid}" if gid is not None else "gpu"
        return "cpu"

    return "cpu"


def _print_eta_pre_fit(model_name: str,
                       resolved_backend: str,
                       wrapper_name: str,
                       params: dict,
                       all_horizons: Sequence[int],
                       run_hpo: bool,
                       eval_horizons: Sequence[int]) -> None:
    """
    Gibt eine grobe, nicht-bindende Laufzeit-Abschätzung in „Work Units“ aus.
    - Trees: Work Units ~ H × n_estimators  (bei HPO zusätzlich × n_trials × n_folds × len(hpo_horizons))
    - DL   : Work Units ~ H × n_epochs      (nur als grobe Orientierung)
    """
    H = len(all_horizons or [])
    rb = (resolved_backend or "").lower()
    mn = (model_name or "").lower()

    # Device-String
    dev_dl = _gpu_name_str()
    dev_tree = _tree_device_label(rb, params)

    # Gemeinsame Kopfzeile
    print(f"[ETA] backend={rb} | wrapper={wrapper_name} | H={H}")

    if rb in ("xgboost", "lightgbm"):
        ne = int(params.get("n_estimators", 0) or 0)
        work_units = H * ne
        print(f"[ETA] device≈{dev_tree} | n_estimators={ne} ⇒ ~{work_units:,} boosting rounds (H × n_estimators)")
        if run_hpo:
            from config import HPO_CFG
            trials = int(HPO_CFG.get("n_trials", 16))
            folds  = int(HPO_CFG.get("n_folds", 3))
            hpo_h  = tuple(HPO_CFG.get("hpo_horizons", eval_horizons or []))
            per_trial_models = len(hpo_h)
            total_fits = trials * folds * per_trial_models
            total_units = work_units * total_fits
            print(f"[ETA] HPO: trials={trials} × folds={folds} × hpo_horizons={per_trial_models}"
                  f" ⇒ ~{total_fits:,} Einzel-Fits")
            print(f"[ETA] HPO Work Units ≈ {work_units:,} × {total_fits:,} = ~{total_units:,} rounds")
        return

    if rb in ("nhits", "tft"):
        # Sanfte Defaults wie in den Model-Buildern
        if rb == "nhits":
            n_epochs = int(params.get("n_epochs", 25))
            bs       = params.get("batch_size", 64)
            ich      = params.get("input_chunk_length", 28)
        else:  # tft
            n_epochs = int(params.get("n_epochs", 30))
            bs       = params.get("batch_size", 128)
            ich      = params.get("input_chunk_length", 56)

        work_units = H * n_epochs
        print(f"[ETA] device≈{dev_dl} | n_epochs={n_epochs} | batch_size={bs} | input_chunk_length={ich}"
              f" ⇒ ~{work_units:,} epoch-units (H × n_epochs)")
        return

    # Fallback (z. B. GNS)
    print(f"[ETA] device≈{dev_dl}")

# =============================================================================
# Helpers
# =============================================================================
def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _resolve_model_classes(model_name: str):
    """Mappt Modellnamen auf passende Wrapper-Klassen und Backend-Label."""
    m = (model_name or "").lower()
    if m in ("xgboost", "xgb"):
        from models_xgb import DirectForecasterXGB, MIMOForecasterXGB
        return DirectForecasterXGB, MIMOForecasterXGB, "xgboost"
    if m in ("nhits", "n-hits"):
        from models_dl import NHiTSForecaster
        return NHiTSForecaster, NHiTSForecaster, "nhits"
    if m in ("tft", "temporal_fusion_transformer"):
        from models_dl import TFTForecaster
        return TFTForecaster, TFTForecaster, "tft"
    if m in ("gns", "global_naive_seasonal"):
        from models_naive import GlobalNaiveSeasonal
        return GlobalNaiveSeasonal, GlobalNaiveSeasonal, "gns"
    # Default → LightGBM
    from models_lgbm import DirectForecaster, MIMOForecaster
    return DirectForecaster, MIMOForecaster, "lightgbm"

def _inject_tree_gpu_params(p: dict, backend: str) -> dict:
    """
    Injiziert GPU-Parameter für XGBoost/LightGBM.
    Aktiviert GPU, wenn:
      - config.FORCE_TREES_ON_GPU=True ODER
      - ENV: XGB_USE_GPU=1 / LGBM_USE_GPU=1
    (ohne Torch-Abhängigkeit)
    """
    from config import FORCE_TREES_ON_GPU, preferred_gpu_id
    q = dict(p or {})

    want_gpu = bool(
        FORCE_TREES_ON_GPU or
        os.getenv("XGB_USE_GPU", "0") == "1" or
        os.getenv("LGBM_USE_GPU", "0") == "1"
    )
    if not want_gpu:
        return q

    gpu_id = preferred_gpu_id()  # nutzt torch falls vorhanden, sonst CUDA_VISIBLE_DEVICES/0
    b = (backend or "").lower()

    if b in ("xgboost", "xgb"):
        try:
            import xgboost as xgb
            from packaging import version as V
            v = V.parse(xgb.__version__)
        except Exception:
            v = None

        if v is not None and v >= V.parse("2.0.0"):
            q["device"] = "cuda"
            q.pop("gpu_id", None)
            q.pop("predictor", None)  # auto ist ok
        else:
            q["tree_method"] = "gpu_hist"
            q["predictor"] = q.get("predictor", "gpu_predictor")
            q["gpu_id"] = gpu_id

    elif b in ("lightgbm", "lgbm"):
        q["device_type"] = "gpu"
        q["gpu_platform_id"] = q.get("gpu_platform_id", 0)
        q["gpu_device_id"] = gpu_id

    return q

def _print_device_summary(model_name: str, resolved_backend: str, params: dict) -> None:
    rb = (resolved_backend or "").lower()
    p = params or {}

    # DL: reale PyTorch-Device-Info
    if rb in ("nhits", "tft"):
        try:
            import torch
            dl_dev = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})" \
                     if torch.cuda.is_available() else "cpu"
        except Exception:
            dl_dev = "cpu"
        print(f"[DEVICE] DL backend={rb} → device={dl_dev}")
        return

    # Trees: aus Parametern ableiten
    if rb in ("xgboost", "xgb"):
        label = p.get("device") or p.get("tree_method") or "cpu"
        pred  = p.get("predictor")
        gid   = p.get("gpu_id")
        print(f"[DEVICE] XGBoost → device/method={label}"
              + (f", predictor={pred}" if pred else "")
              + (f", gpu_id={gid}" if gid is not None else ""))
        return

    if rb in ("lightgbm", "lgbm"):
        dtype = str(p.get("device_type", "cpu")).lower()
        gid   = p.get("gpu_device_id")
        print(f"[DEVICE] LightGBM → device_type={dtype}"
              + (f", gpu_device_id={gid}" if gid is not None else ""))
        return

    print(f"[DEVICE] backend={rb} → device=cpu")


def _build_init_kwargs(Cls, params, train_horizons, max_h, is_mimo: bool) -> dict:
    """Stellt die korrekten __init__-Argumente für die jeweilige Wrapper-Klasse zusammen."""
    sig = inspect.signature(Cls.__init__)
    kw = {}

    # Zeitachsen
    if "horizons" in sig.parameters:
        kw["horizons"] = train_horizons
    elif "H" in sig.parameters:
        kw["H"] = int(max_h)
    elif "output_chunk_length" in sig.parameters:
        kw["output_chunk_length"] = int(max_h)

    # Hyperparameter-Container
    for pname in ("model_params", "lgbm_params", "xgb_params", "params", "dl_params"):
        if pname in sig.parameters:
            kw[pname] = params
            break

    # MIMO-spezifisch
    if is_mimo and "mor_n_jobs" in sig.parameters:
        kw["mor_n_jobs"] = 1

    return kw


def _safe_num(x):
    """Konvertiert numerische Werte robust für W&B-Summary."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    try:
        return float(x)
    except Exception:
        return None


# =============================================================================
# Runner
# =============================================================================
def run_training(
    df_raw: pd.DataFrame,
    df_name: str,
    model_name: str,
    wrapper: str = "direct",
    *,
    max_h: int,
    eval_horizons: Sequence[int],
    n_test_origins: int = 1,
    run_name_suffix: str = "",
    log_wandb: bool = USE_WANDB,
    seasonal_period: int = SEASONAL_PERIOD,
    model_params: Dict | None = None,
    feature_cols: Sequence[str] | None = None,
    shap_sample_rows: int = 1024,
    use_hptuned_params: bool = False,
    run_hpo: bool = False,
    subset: bool = False,
) -> Dict:
    """End-to-End Training + Evaluation inkl. optionalem HPO und W&B-Logging."""
    _log("---- Forecasting Pipeline: run_training() ----")
    print("use_hptuned_params =", use_hptuned_params, ", USE_HPTUNED_PARAMS =", USE_HPTUNED_PARAMS)

    # ------------------------------------------------------------------ #
    # 0) Reproduzierbarkeit & Eingaben prüfen
    # ------------------------------------------------------------------ #
    set_global_seed(RANDOM_STATE)

    df_name = str(df_name).lower()
    allowed_datasets = ("bakery", "m5")
    if df_name not in allowed_datasets:
        raise ValueError(f"df_name must be one of {allowed_datasets}, got '{df_name}'")

    allowed_wrappers = ("direct", "mimo", "baseline", "native")
    wrapper_original = str(wrapper)
    alias_map = {"native": "mimo", "baseline": "mimo"}
    wrapper_fit = alias_map.get(wrapper_original.lower(), wrapper_original.lower())
    if wrapper_fit not in ("direct", "mimo"):
        raise ValueError(f"wrapper must be one of ('direct','mimo','native','baseline'), got '{wrapper}'")

    model_name = str(model_name).lower()
    allowed_models = ("lightgbm", "xgboost", "nhits", "tft", "gns")
    if model_name not in allowed_models:
        raise ValueError(f"model_name must be one of {allowed_models}, got '{model_name}'")

    # ------------------------------------------------------------------ #
    # 1) Datenaufbereitung
    # ------------------------------------------------------------------ #
    dataset_spec = get_spec(df_name)
    df = dataset_spec.normalize(df_raw)
    df = sort_df(df)
    df = cast_categoricals(df)

    # Ziele (immer 1..max_h)
    all_horizons = tuple(range(1, int(max_h) + 1))
    df_tgt = make_horizon_targets(df, TARGET_COL, all_horizons, assume_sorted=True)

    # Split (Origin-basiert)
    train_df, test_df = time_based_origin_split(
        df_tgt, test_days=int(n_test_origins), max_h=max_h, assume_sorted=True
    )

    # Meta-Zahlen
    n_series_all = count_series(df)
    n_series_train = count_series(train_df)
    n_series_test = count_series(test_df)

    # Features
    feat_cols = list(feature_cols) if feature_cols is not None else dataset_spec.feature_columns(df_tgt)

    # RMSSE-Skalen
    scales = rmsse_scales(train_df, target_col=TARGET_COL, seasonal_period=seasonal_period)

    # ------------------------------------------------------------------ #
    # 2) Modellwahl & Hyperparameter
    # ------------------------------------------------------------------ #
    wrapper_name = wrapper_fit
    eval_horizons = tuple(int(h) for h in eval_horizons)

    # Train-Horizonte: für Vergleichbarkeit beide Wraps über 1..max_h trainieren
    train_horizons = all_horizons

    DirectCls, MIMOCls, _resolved = _resolve_model_classes(model_name)

    # Fallback-Defaults je Backend
    if _resolved == "lightgbm":
        default_fallback = dict(LGBM_PARAMS)
    elif _resolved == "xgboost":
        default_fallback = dict(XGB_PARAMS)
    elif _resolved == "nhits":
        default_fallback = dict(NHITS_PARAMS)
    elif _resolved == "tft":
        default_fallback = dict(TFT_PARAMS)
    elif _resolved == "gns":
        default_fallback = dict(NAIVE_PARAMS)
    else:
        default_fallback = {}

    used_tuned_params = False
    param_source = "default"
    params: Dict

    if model_params is not None:
        # 2a) Explizit übergebene Parameter
        params = dict(model_params)
        param_source = "explicit"
    else:
        # 2b) Optionales HPO (nur sinnvoll für MIMO)
        best_from_hpo = None
        if run_hpo and wrapper_name == "mimo":
            # kleine HPO-Vorankündigung
            try:
                trials = int(HPO_CFG.get("n_trials", 16))
                folds  = int(HPO_CFG.get("n_folds", 3))
                hpo_h  = tuple(HPO_CFG.get("hpo_horizons", eval_horizons))
                per_trial_models = len(hpo_h)
                print(f"[HPO] Starting Optuna: trials={trials}, folds={folds}, hpo_horizons={list(hpo_h)} "
                      f"(≈ {trials * folds * per_trial_models} Einzel-Fits)")
            except Exception:
                pass
            
            # HPO-Zielfunktion
            def train_and_score(p: dict, tr: pd.DataFrame, va: pd.DataFrame) -> float:
                """HPO-Zielfunktion: trainiert auf TR, evaluiert RMSSE auf VA."""
                tr_scales = rmsse_scales(tr, target_col=TARGET_COL, seasonal_period=seasonal_period)
                hpo_horizons = tuple(HPO_CFG.get("hpo_horizons", eval_horizons))
                y_cols_used_hpo = [f"y+{h}" for h in hpo_horizons]

                X_tr = tr.loc[:, feat_cols].to_numpy(copy=False)
                X_va = va.loc[:, feat_cols].to_numpy(copy=False)
                Y_tr = tr.loc[:, y_cols_used_hpo].to_numpy(copy=False)
                Y_va = va.loc[:, y_cols_used_hpo].to_numpy(copy=False)

                esr = int(HPO_CFG.get("early_stopping_rounds", 0))
                preds_va = np.zeros_like(Y_va, dtype=float)
                backend = (_resolved or "").lower()

                # ---- XGBoost ------------------------------------------------
                if backend == "xgboost":
                    if XGBRegressor is None or xgb is None:
                        raise ImportError("XGBoost nicht verfügbar – bitte `pip install xgboost`.")

                    fit_sig = inspect.signature(XGBRegressor.fit)
                    supports_callbacks = "callbacks" in fit_sig.parameters
                    supports_esr = "early_stopping_rounds" in fit_sig.parameters

                    p_gpu = _inject_tree_gpu_params(p, backend="xgboost")
                    if _resolved in ("xgboost", "lightgbm"):
                        dev_label = p_gpu.get("device") or p_gpu.get("tree_method") or p_gpu.get("device_type")
                        print(f"[DEVICE] {_resolved}: {dev_label} | CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','unset')}")
                        if _resolved == "lightgbm":
                            print(f"[DEVICE] lightgbm: platform_id={p_gpu.get('gpu_platform_id')} device_id={p_gpu.get('gpu_device_id')}")

                    for j, _h in enumerate(hpo_horizons):
                        est = XGBRegressor(**_sanitize_xgb_params(p_gpu))
                        if esr > 0:
                            if supports_callbacks and hasattr(xgb, "callback"):
                                cb = [xgb.callback.EarlyStopping(rounds=int(esr), save_best=True, maximize=False)]
                                est.fit(X_tr, Y_tr[:, j], eval_set=[(X_va, Y_va[:, j])], callbacks=cb, verbose=False)
                                preds_va[:, j] = est.predict(X_va)
                            elif supports_esr:
                                est.fit(X_tr, Y_tr[:, j], eval_set=[(X_va, Y_va[:, j])],
                                        early_stopping_rounds=int(esr), verbose=False)
                                best_it = getattr(est, "best_iteration", None)
                                try:
                                    preds_va[:, j] = (
                                        est.predict(X_va, iteration_range=(0, int(best_it) + 1))
                                        if best_it is not None else
                                        est.predict(X_va, ntree_limit=int(getattr(est, "best_ntree_limit", 0)) or None)
                                    )
                                except TypeError:
                                    preds_va[:, j] = est.predict(X_va, ntree_limit=int(getattr(est, "best_ntree_limit", 0)) or None)
                            else:
                                est.fit(X_tr, Y_tr[:, j], verbose=False)
                                preds_va[:, j] = est.predict(X_va)
                        else:
                            est.fit(X_tr, Y_tr[:, j], verbose=False)
                            preds_va[:, j] = est.predict(X_va)

                # ---- LightGBM -----------------------------------------------
                elif backend == "lightgbm":
                    if LGBMRegressor is None or lgbm is None:
                        raise ImportError("LightGBM nicht verfügbar – bitte `pip install lightgbm`.")
                    
                    p_gpu = _inject_tree_gpu_params(p, backend="lightgbm")
                    if _resolved in ("xgboost", "lightgbm"):
                        dev_label = p_gpu.get("device") or p_gpu.get("tree_method") or p_gpu.get("device_type")
                        print(f"[DEVICE] {_resolved}: {dev_label} | CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','unset')}")
                        
                        if _resolved == "lightgbm":
                            print(f"[DEVICE] lightgbm: platform_id={p_gpu.get('gpu_platform_id')} device_id={p_gpu.get('gpu_device_id')}")
                    
                    def _strip_lgbm_gpu_local(q: dict) -> dict:
                        r = dict(q)
                        for k in ("device_type", "gpu_platform_id", "gpu_device_id"):
                            r.pop(k, None)
                        return r

                    for j, _h in enumerate(hpo_horizons):
                        params_try = dict(p_gpu)
                        try:
                            est = LGBMRegressor(**params_try)
                            if esr > 0 and early_stopping is not None:
                                est.fit(
                                    X_tr, Y_tr[:, j],
                                    eval_set=[(X_va, Y_va[:, j])],
                                    callbacks=[early_stopping(stopping_rounds=esr, verbose=False)],
                                )
                                preds_va[:, j] = est.predict(X_va, num_iteration=getattr(est, "best_iteration_", None))
                            else:
                                est.fit(X_tr, Y_tr[:, j])
                                preds_va[:, j] = est.predict(X_va)
                        except Exception as e:
                            msg = str(e)
                            if ("GPU" in msg) or ("gpu" in msg) or ("Device" in msg):
                                params_cpu = _strip_lgbm_gpu_local(params_try)
                                est = LGBMRegressor(**params_cpu)
                                if esr > 0 and early_stopping is not None:
                                    est.fit(
                                        X_tr, Y_tr[:, j],
                                        eval_set=[(X_va, Y_va[:, j])],
                                        callbacks=[early_stopping(stopping_rounds=esr, verbose=False)],
                                    )
                                    preds_va[:, j] = est.predict(X_va, num_iteration=getattr(est, "best_iteration_", None))
                                else:
                                    est.fit(X_tr, Y_tr[:, j])
                                    preds_va[:, j] = est.predict(X_va)
                            else:
                                raise

                # ---- DL (Darts) ---------------------------------------------
                elif backend in ("nhits", "tft"):
                    DirectDL, MIMODL, _ = _resolve_model_classes(model_name)
                    DLCls = MIMODL if wrapper_name == "mimo" else DirectDL
                    mdl = DLCls(horizons=hpo_horizons, model_params=p)
                    mdl.fit(tr, feature_cols=feat_cols)
                    Yp_df = mdl.predict(va)
                    Yp = Yp_df.loc[:, y_cols_used_hpo]
                    Yt = va.loc[:, y_cols_used_hpo]
                    series_idx = (
                        pd.Index(va[SERIES_COLS[0]]) if len(SERIES_COLS) == 1
                        else pd.MultiIndex.from_frame(va.loc[:, list(SERIES_COLS)])
                    )
                    _, rmsse_total_hpo = rmsse_per_h_and_total(Yt, Yp, series_idx, tr_scales)
                    return float(rmsse_total_hpo)

                # ---- Naive ---------------------------------------------------
                elif backend in ("gns", "naive"):
                    from models_naive import GlobalNaiveSeasonal
                    mdl = GlobalNaiveSeasonal(horizons=hpo_horizons, model_params=p)
                    mdl.fit(tr, feature_cols=feat_cols)
                    Yp_df = mdl.predict(va)
                    Yp = Yp_df.loc[:, y_cols_used_hpo]
                    Yt = va.loc[:, y_cols_used_hpo]
                    series_idx = (
                        pd.Index(va[SERIES_COLS[0]]) if len(SERIES_COLS) == 1
                        else pd.MultiIndex.from_frame(va.loc[:, list(SERIES_COLS)])
                    )
                    _, rmsse_total_hpo = rmsse_per_h_and_total(Yt, Yp, series_idx, tr_scales)
                    return float(rmsse_total_hpo)

                else:
                    raise ValueError(f"Unknown backend in HPO: {backend}")

                # Gemeinsame RMSSE-Berechnung (Trees)
                Yp = pd.DataFrame(preds_va, index=va.index, columns=y_cols_used_hpo)
                Yt = va.loc[:, y_cols_used_hpo]
                series_idx = (
                    pd.Index(va[SERIES_COLS[0]]) if len(SERIES_COLS) == 1
                    else pd.MultiIndex.from_frame(va.loc[:, list(SERIES_COLS)])
                )
                _, rmsse_total_hpo = rmsse_per_h_and_total(Yt, Yp, series_idx, tr_scales)
                return float(rmsse_total_hpo)

            best_from_hpo = run_optuna_hpo(
                dataset_name=df_name,
                train_df=train_df,
                max_h=max_h,
                train_and_score_fn=train_and_score,
                model_name=model_name,
                wrapper=wrapper_name,
            )

        # Finale Parameterauswahl (HPO-Ergebnis / gespeicherte / Defaults)
        if run_hpo and wrapper_name == "mimo" and best_from_hpo is not None and (USE_HPTUNED_PARAMS and use_hptuned_params):
            params = dict(best_from_hpo)
            used_tuned_params = True
            param_source = "fresh_hpo"
        else:
            params, used_tuned_params = get_params_for_dataset(
                model_name=model_name,
                wrapper=wrapper_name,
                dataset_name=df_name,
                allow_tuned=(USE_HPTUNED_PARAMS and use_hptuned_params),
                fallback=default_fallback,
            )
            param_source = "saved" if used_tuned_params else "default"

        _log(f"[PARAMS] source={param_source} | USE_HPTUNED_PARAMS={USE_HPTUNED_PARAMS}, use_hptuned_params={use_hptuned_params}")

    # Vor Instanziierung: zentrale GPU-Injektion für Tree-Backends
    if _resolved in ("xgboost", "lightgbm"):
        params = _inject_tree_gpu_params(params, backend=_resolved)
        
    if _resolved in ("xgboost", "lightgbm"):
        dev_label = params.get("device") or params.get("tree_method") or params.get("device_type")
        print(f"[DEVICE] {_resolved}: {dev_label} | CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','unset')}")
        if _resolved == "lightgbm":
            print(f"[DEVICE] lightgbm: platform_id={params.get('gpu_platform_id')} device_id={params.get('gpu_device_id')}")

    # ------------------------------------------------------------------ #
    # 3) Modell instanziieren & trainieren
    # ------------------------------------------------------------------ #
    if wrapper_name == "direct":
        init_kwargs = _build_init_kwargs(DirectCls, params, train_horizons, max_h, is_mimo=False)
        model = DirectCls(**init_kwargs)
    elif wrapper_name == "mimo":
        init_kwargs = _build_init_kwargs(MIMOCls, params, train_horizons, max_h, is_mimo=True)
        model = MIMOCls(**init_kwargs)
    else:
        raise ValueError("wrapper muss 'direct' oder 'mimo' sein")

    _log("3) Starting training")
    _log("[PARAMS USED] "
         f"n_estimators={params.get('n_estimators')} | "
         f"max_depth={params.get('max_depth')} | "
         f"min_child_weight={params.get('min_child_weight')} | "
         f"subsample={params.get('subsample')} | "
         f"colsample_bytree={params.get('colsample_bytree')} | "
         f"reg_alpha={params.get('reg_alpha')} | "
         f"reg_lambda={params.get('reg_lambda')}")
    
    # Vor dem Training ETA-Schätzung & Device-Zusammenfassung ausgeben
    _print_device_summary(model_name=model_name, resolved_backend=_resolved, params=params)
    _print_eta_pre_fit(
        model_name=model_name,
        resolved_backend=_resolved,
        wrapper_name=wrapper_name,
        params=params,
        all_horizons=all_horizons,
        run_hpo=bool(run_hpo),
        eval_horizons=eval_horizons,
    )


    # Ressourcen-Messung (vor Training)
    proc = psutil.Process(os.getpid()) if psutil is not None else None
    cpu_t0 = proc.cpu_times() if proc else None
    rss0 = (proc.memory_info().rss if proc else None)
    t0 = time.perf_counter()

    # Fit
    model.fit(train_df, feature_cols=feat_cols)

    # Ressourcen-Messung (nach Training)
    t1 = time.perf_counter()
    cpu_t1 = proc.cpu_times() if proc else None
    rss1 = (proc.memory_info().rss if proc else None)

    # Peak-RSS (best effort)
    peak_rss_mb = None
    if resource is not None:
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            val = float(getattr(ru, "ru_maxrss", 0.0))
            peak_rss_mb = (val / (1024**2)) if val > 10**9 else ((val * 1024.0) / (1024**2))
        except Exception:
            peak_rss_mb = None

    wall = max(t1 - t0, 1e-9)
    cpu_user = (cpu_t1.user - cpu_t0.user) if (cpu_t0 and cpu_t1) else float("nan")
    cpu_sys = (cpu_t1.system - cpu_t0.system) if (cpu_t0 and cpu_t1) else float("nan")
    cpu_tot = (cpu_user + cpu_sys) if (cpu_user == cpu_user and cpu_sys == cpu_sys) else float("nan")

    # Parallelitäts-Metriken
    try:
        n_cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        n_cores = 1
    cpu_wall_ratio = (cpu_tot / wall) if (cpu_tot == cpu_tot) else float("nan")
    core_util = min(1.0, cpu_wall_ratio / n_cores) if (cpu_wall_ratio == cpu_wall_ratio) else float("nan")

    n_estimators = params.get("n_estimators") if isinstance(params, dict) else None
    horizons_cnt = len(all_horizons) if isinstance(all_horizons, (list, tuple)) else None

    train_profile = {
        "train_wall_time_s": wall,
        "train_cpu_time_user_s": cpu_user,
        "train_cpu_time_system_s": cpu_sys,
        "train_cpu_time_total_s": cpu_tot,
        "train_cpu_wall_ratio": cpu_wall_ratio,
        "train_core_utilization": core_util,
        "n_logical_cores": int(n_cores),
        "n_jobs_param": int(params.get("n_jobs")) if isinstance(params, dict) and "n_jobs" in params else None,
        "horizons_count": int(horizons_cnt) if horizons_cnt is not None else None,
        "n_estimators": int(n_estimators) if n_estimators is not None else None,
        "peak_rss_mb": peak_rss_mb if peak_rss_mb is not None else float("nan"),
        "rss_end_mb": (rss1 / (1024**2)) if rss1 is not None else float("nan"),
    }
    force_summary = {"train_cpu_time_total_s": _safe_num(train_profile.get("train_cpu_time_total_s"))}

    # ------------------------------------------------------------------ #
    # 4) Vorhersage & Metriken
    # ------------------------------------------------------------------ #
    _log("4) Starting prediction + evaluation")
    Y_pred_full = model.predict(test_df)

    y_cols_used = [f"y+{h}" for h in all_horizons]
    Y_true_used = test_df.loc[:, y_cols_used]
    Y_pred_used = Y_pred_full.loc[:, y_cols_used]

    series_index = (
        pd.Index(test_df[SERIES_COLS[0]]) if len(SERIES_COLS) == 1
        else pd.MultiIndex.from_frame(test_df.loc[:, list(SERIES_COLS)])
    )
    rmsse_h, rmsse_total = rmsse_per_h_and_total(Y_true_used, Y_pred_used, series_index, scales)
    _log(f"RMSSE@h={sorted(rmsse_h.keys())} | total={rmsse_total:.6f}")

    per_series_rmsse = rmsse_per_series_df(
        Y_true=Y_true_used,
        Y_pred=Y_pred_used,
        series_index=series_index,
        scales=scales,
        series_cols=list(SERIES_COLS),
        keys_df=test_df,
    )

    extra_totals = regression_metrics_total(Y_true_used, Y_pred_used)
    extra_table = regression_metrics_table(Y_true_used, Y_pred_used)

    compute_backend, compute_backend_detail = detect_compute_backend(model_name, model, params)
    _log(f"[Compute] backend={compute_backend} ({compute_backend_detail})")

    log_extras = not run_hpo
    fi = feature_importance_df(model, feat_cols) if log_extras else None

    # ------------------------------------------------------------------ #
    # 5) W&B Logging (optional)
    # ------------------------------------------------------------------ #
    _log("5) Starting prediction + evaluation")
    run_ref = None
    if log_wandb:
        _log("---- W&B Logging ----")
        used_eval_list = [int(str(c).split("+", 1)[1]) for c in y_cols_used]
        ctx = wandb_run_context(
            model_name=model_name, wrapper=wrapper_original, dataset_name=df_name, subset=subset,
            eval_horizons=used_eval_list, n_test_origins=int(n_test_origins),
            use_hptuned_params=use_hptuned_params, run_hpo=run_hpo,
            used_tuned_params=used_tuned_params, params_source=param_source,
            model_params=params,
        )
        run_ref = init_run(
            ctx,
            run_name_suffix=run_name_suffix,
            horizons=used_eval_list,
            test_days=int(n_test_origins),
            train_df=train_df,
            test_df=test_df,
            feat_cols=feat_cols,
            model_params=params,
            n_series_total=n_series_all,
            n_series_train=n_series_train,
            n_series_test=n_series_test,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            subset=subset,
        )
        if run_ref is not None:
            for k, v in force_summary.items():
                if v is not None:
                    run_ref.summary[k] = v
            run_ref.summary["compute_backend"] = compute_backend
            run_ref.summary["compute_backend_detail"] = compute_backend_detail

            if model_name in ("xgboost", "xgb"):
                run_ref.summary.update({
                    "xgb_device_or_method": params.get("device") or params.get("tree_method"),
                    "xgb_predictor": params.get("predictor"),
                    "xgb_gpu_id": params.get("gpu_id"),
                })
            if model_name in ("nhits", "tft"):
                try:
                    import torch
                    run_ref.summary["dl_device"] = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    pass

            run_ref.config.update({
                "train_horizons": list(map(int, train_horizons)),
                "used_tuned_params": bool(used_tuned_params),
                "param_source": param_source,
                "feature_cols_count": len(feat_cols),
            }, allow_val_change=True)

            print("W&B URL:", getattr(run_ref, "url", None))

        # Metriken & Artefakte
        log_metrics(run_ref, rmsse_total, rmsse_h)
        for h, v in sorted(rmsse_h.items()):
            run_ref.log({"rmsse_by_h": float(v)}, step=int(h))

        log_regression_metrics(run_ref, extra_totals, extra_table)
        log_train_profile(run_ref, train_profile)

        if log_extras:
            log_feature_importance(run_ref, fi, top_k=25)
            if LOG_SHAP and shap_sample_rows > 0:
                shap_df = shap_summary_df(model, train_df, feat_cols, sample_rows=shap_sample_rows)
                if shap_df is not None and not shap_df.empty:
                    log_shap_summary(run_ref, shap_df, top_k=25)
                X_rep, shap_vals = shap_values_aggregate(model, train_df, feat_cols, sample_rows=shap_sample_rows)
                if shap_vals.size > 0:
                    outdir = getattr(run_ref, "dir", os.getenv("WANDB_DIR", ".")) or "."
                    outpath = shap_beeswarm_figure(shap_vals, X_rep, max_display=25, outdir=outdir,
                                                   filename="shap_beeswarm_allH.png")
                    _log(f"[SHAP] Beeswarm saved to: {outpath}")
                    if outpath:
                        log_image(run_ref, "shap_beeswarm_allH", outpath)

            log_test_sample(run_ref, SERIES_COLS, DATE_COL, test_df, Y_true_used, Y_pred_used, max_rows=200)
            if per_series_rmsse is not None and not per_series_rmsse.empty:
                log_rmsse_per_series(run_ref, per_series_rmsse, top_n=100)

        finish_run(run_ref)

    # ------------------------------------------------------------------ #
    # 6) Console-Summary & Rückgabe
    # ------------------------------------------------------------------ #
    key_cols = list(SERIES_COLS) + [DATE_COL]
    key_cols += [c for c in AUX_KEY_COLS if c in test_df.columns]

    eval_span = f"{min(all_horizons)}..{max(all_horizons)}" if len(all_horizons) > 6 else ",".join(map(str, all_horizons))
    print(
        "[DONE] "
        f"{df_name} | {model_name}/{wrapper_original} | "
        f"H={max_h} (eval:{eval_span}) | "
        f"test_origins={n_test_origins} | "
        f"series={n_series_train}/{n_series_test}/{n_series_all} | "
        f"features={len(feat_cols)} | "
        f"RMSSE_total={rmsse_total:.4f} | "
        f"{compute_backend}:{compute_backend_detail}"
    )

    return {
        "model": model,
        "wrapper": wrapper_original,
        "feature_cols": feat_cols,
        "rmsse_total": float(rmsse_total),
        "rmsse_by_horizon": {int(k): float(v) for k, v in rmsse_h.items()},
        "test_predictions": Y_pred_used,
        "test_truth": Y_true_used,
        "test_keys": test_df.loc[:, list(SERIES_COLS) + [DATE_COL]],
        "train_profile": train_profile,
        "extra_metrics_total": extra_totals,
        "extra_metrics_by_horizon": extra_table,
        "feature_importance": fi,
        "wandb_run": run_ref,
        "params": params,
        "param_source": param_source,
        "used_tuned_params": used_tuned_params,
        "rmsse_per_series": per_series_rmsse,
        "test_keys": test_df.loc[:, key_cols],
        "compute_backend": compute_backend,
        "compute_backend_detail": compute_backend_detail,
    }
    
# Ende file pipeline.py 
