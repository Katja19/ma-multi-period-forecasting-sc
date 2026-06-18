# hpo.py
from __future__ import annotations

"""
Hyperparameter-Optimierung (HPO) – zentrale Utilities und Konfiguration.
- Saubere Import-Gruppierung (Stdlib / Third-Party / Local)
- Optionale Dependencies (Torch) mit Flag
- Konsistente Param-Imports mit robusten Fallbacks
"""

# ---------------------------- Standardbibliothek ----------------------------
import json
from copy import deepcopy
from typing import Callable, Literal, Optional, List

# ------------------------------ Third-Party --------------------------------
import numpy as np
import pandas as pd

# --------------------------------- Local -----------------------------------
from core import rolling_train_val_windows, sort_df

# Harte Abhängigkeiten aus config
from config import (
    HPO_CFG,
    RANDOM_STATE,
    AUTO_SAVE_HPO,
    save_hptuned_params,
    FORCE_TREES_ON_GPU,
    preferred_gpu_id,
)

# Weiche Abhängigkeiten aus config (Fallbacks auf leere Dicts)
try:
    from config import (
        LGBM_PARAMS,
        XGB_PARAMS,
        NHITS_PARAMS,
        TFT_PARAMS,
        NAIVE_PARAMS,
    )
except Exception:  # pragma: no cover
    LGBM_PARAMS: dict = {}
    XGB_PARAMS: dict = {}
    NHITS_PARAMS: dict = {}
    TFT_PARAMS: dict = {}
    NAIVE_PARAMS: dict = {}

# Optional: Torch nur für DL-Modelle / GPU-Check
try:
    import torch  # type: ignore
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAVE_TORCH = False


# ------------------------------------------------------------
# GPU helper for DL trials
# ------------------------------------------------------------
def _dl_trainer_kwargs():
    use_cuda = bool(torch is not None and torch.cuda.is_available())
    return dict(accelerator="gpu", devices=1, enable_progress_bar=False) if use_cuda else \
           dict(accelerator="cpu", enable_progress_bar=False)

def suggest_gns_params(trial) -> dict:
    # K = Anzahl vergangener Saisons zum Mittel; P bleibt fix (7)
    return dict(
        K=trial.suggest_int("K", 1, 28),   # typischer Wochenzyklus; gern anpassen (z.B. 1..14)
        seasonal_period=7 # fix damit vergleichbar bleibt mit anderen Modellen
    )
    
# ------------------------------------------------------------
# Search Space: LightGBM  (GPU-ready via config)
# ------------------------------------------------------------
def suggest_lgbm_params(trial) -> dict:
    p = {}
    # --- FAST: kompakter & schneller ---
    p["n_estimators"]      = trial.suggest_int("n_estimators", 300, 900, step=100)
    p["learning_rate"]     = trial.suggest_float("learning_rate", 0.06, 0.15, log=True)
    p["num_leaves"]        = trial.suggest_int("num_leaves", 48, 128, step=16)
    p["max_depth"]         = trial.suggest_int("max_depth", 4, 10, step=2)
    p["min_child_samples"] = trial.suggest_int("min_child_samples", 40, 120, step=20)
    p["subsample"]         = trial.suggest_float("subsample", 0.7, 0.95)
    p["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.7, 1.0)
    p["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-2, 2.0, log=True)
    p["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-4, 0.5, log=True)
    p["max_bin"]           = trial.suggest_int("max_bin", 127, 191)

    # Projekt-Defaults
    p["random_state"]      = RANDOM_STATE
    p["n_jobs"]            = LGBM_PARAMS.get("n_jobs", 6)
    p["verbosity"]         = LGBM_PARAMS.get("verbosity", -1)
    p["force_col_wise"]    = True
    p["bagging_freq"]      = 1

    # GPU aus config übernehmen …
    if "device_type" in LGBM_PARAMS:
        p["device_type"] = LGBM_PARAMS.get("device_type")
        for k in ("gpu_platform_id", "gpu_device_id"):
            if k in LGBM_PARAMS:
                p[k] = LGBM_PARAMS[k]
    # … oder GPU hart setzen, wenn erzwingen erlaubt & verfügbar
    if FORCE_TREES_ON_GPU and HAVE_TORCH and torch.cuda.is_available():
        p.setdefault("device_type", "gpu")
        p.setdefault("gpu_device_id", preferred_gpu_id())
        p.setdefault("gpu_platform_id", 0)
    return p
# ------------------------------------------------------------
# ausführlichere Suche (mehr Trees, kleinere LR): Alternativen (nicht löschen)
# def suggest_lgbm_params(trial) -> dict:
#     p = {}
#     # moderat größere Kapazität, kleinere LR (GPU)
#     p["n_estimators"]      = trial.suggest_int("n_estimators", 600, 1600, step=200)
#     p["learning_rate"]     = trial.suggest_float("learning_rate", 0.02, 0.08, log=True)
#     p["num_leaves"]        = trial.suggest_int("num_leaves", 64, 256, step=16)
#     p["max_depth"]         = trial.suggest_int("max_depth", 6, 12, step=2)
#     p["min_child_samples"] = trial.suggest_int("min_child_samples", 20, 120, step=10)
#     p["subsample"]         = trial.suggest_float("subsample", 0.7, 1.0)
#     p["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.7, 1.0)
#     p["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True)
#     p["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
#     p["max_bin"]           = trial.suggest_int("max_bin", 191, 255)

#     # Projekt-Defaults
#     p["random_state"]      = RANDOM_STATE
#     p["n_jobs"]            = LGBM_PARAMS.get("n_jobs", 6)
#     p["verbosity"]         = LGBM_PARAMS.get("verbosity", -1)   # LightGBM: -1 silent
#     p["force_col_wise"]    = True
#     p["bagging_freq"]      = 1  # stabil & schnell, keine Extra-Suche

#     # GPU-Keys: aus config übernehmen …
#     if "device_type" in LGBM_PARAMS:
#         p["device_type"] = LGBM_PARAMS.get("device_type")
#         for k in ("gpu_platform_id", "gpu_device_id"):
#             if k in LGBM_PARAMS:
#                 p[k] = LGBM_PARAMS[k]
#     # … oder hart setzen, wenn FORCE_TREES_ON_GPU aktiv ist
#     if FORCE_TREES_ON_GPU and HAVE_TORCH and torch.cuda.is_available():
#         p.setdefault("device_type", "gpu")
#         p.setdefault("gpu_device_id", preferred_gpu_id())
#         p.setdefault("gpu_platform_id", 0)

#     return p


# ------------------------------------------------------------
# Search Space: XGBoost (GPU via config.XGB_PARAMS)
# ------------------------------------------------------------
# FAST Variante (kompakter & schneller)
def suggest_xgb_params(trial) -> dict:
    p = {}
    # --- FAST: weniger Bäume, flacher, gröbere Bins ---
    p["n_estimators"]      = trial.suggest_int("n_estimators", 300, 900, step=150)
    p["learning_rate"]     = trial.suggest_float("learning_rate", 0.06, 0.15, log=True)
    p["max_depth"]         = trial.suggest_int("max_depth", 4, 8, step=2)
    p["min_child_weight"]  = trial.suggest_float("min_child_weight", 2.0, 6.0)
    p["subsample"]         = trial.suggest_float("subsample", 0.7, 0.95)
    p["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.7, 1.0)
    p["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-2, 2.0, log=True)
    p["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-4, 0.5, log=True)
    p["gamma"]             = trial.suggest_float("gamma", 0.0, 1.0)
    p["max_bin"]           = trial.suggest_categorical("max_bin", [128, 256])

    # Projekt-Defaults
    p["random_state"]      = RANDOM_STATE
    p["n_jobs"]            = XGB_PARAMS.get("n_jobs", 6)
    p["verbosity"]         = XGB_PARAMS.get("verbosity", 0)
    p["tree_method"]       = XGB_PARAMS.get("tree_method", "hist")
    p["predictor"]         = XGB_PARAMS.get("predictor", "auto")
    p["objective"]         = "reg:squarederror"

    # GPU-Keys aus config übernehmen (falls vorhanden)
    if "device" in XGB_PARAMS:
        p["device"] = XGB_PARAMS["device"]
    for k in ("gpu_id", "single_precision_histogram"):
        v = XGB_PARAMS.get(k)
        if v is not None:
            p[k] = v

    # … oder GPU erzwingen (XGB ≥2.0 versteht device="cuda")
    if FORCE_TREES_ON_GPU and HAVE_TORCH and torch.cuda.is_available():
        p["device"] = "cuda"
        p.setdefault("gpu_id", preferred_gpu_id())
        p.setdefault("single_precision_histogram", True)
    return p

# Ausfühhrlichere Suche (mehr Trees, kleinere LR): Alternativen (nicht löschen)
# def suggest_xgb_params(trial) -> dict:
#     p = {}
#     # moderat mehr Trees + kleinere LR (GPU), sonst kompakt halten
#     p["n_estimators"]      = trial.suggest_int("n_estimators", 600, 1400, step=200)
#     p["learning_rate"]     = trial.suggest_float("learning_rate", 0.02, 0.08, log=True)
#     p["max_depth"]         = trial.suggest_int("max_depth", 6, 12, step=2)
#     p["min_child_weight"]  = trial.suggest_float("min_child_weight", 1.0, 8.0, log=True)
#     p["subsample"]         = trial.suggest_float("subsample", 0.7, 1.0)
#     p["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.7, 1.0)
#     p["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True)
#     p["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
#     p["gamma"]             = trial.suggest_float("gamma", 0.0, 2.0)  # stabilisiert Splits
#     p["max_bin"]           = trial.suggest_categorical("max_bin", [256, 384, 512])

#     # Projekt-Defaults
#     p["random_state"]      = RANDOM_STATE
#     p["n_jobs"]            = XGB_PARAMS.get("n_jobs", 6)
#     p["verbosity"]         = XGB_PARAMS.get("verbosity", 0)     # 0 silent
#     p["tree_method"]       = XGB_PARAMS.get("tree_method", "hist")
#     p["predictor"]         = XGB_PARAMS.get("predictor", "auto")
#     p["objective"]         = "reg:squarederror"

#     # GPU-Keys aus config übernehmen
#     if "device" in XGB_PARAMS:
#         p["device"] = XGB_PARAMS["device"]
#     for k in ("gpu_id", "single_precision_histogram"):
#         v = XGB_PARAMS.get(k)
#         if v is not None:
#             p[k] = v

#     # … oder GPU erzwingen, wenn FORCE_TREES_ON_GPU aktiv ist
#     if FORCE_TREES_ON_GPU and HAVE_TORCH and torch.cuda.is_available():
#         # XGB >= 2.0: 'device="cuda"' wird im Modell-Sanitizer korrekt behandelt
#         p["device"] = "cuda"
#         p.setdefault("gpu_id", preferred_gpu_id())
#         # schneller auf vielen Karten, Genauigkeit bleibt für HPO ok
#         p.setdefault("single_precision_histogram", True)

#     return p


# ------------------------------------------------------------
# Search Space: N-HiTS (Darts / PyTorch)
# ------------------------------------------------------------
def suggest_nhits_params(trial) -> dict:
    p = {}
    p["input_chunk_length"] = trial.suggest_int("input_chunk_length", 32, 112, step=8)
    p["dropout"]            = trial.suggest_float("dropout", 0.0, 0.3)
    p["n_epochs"]           = trial.suggest_int("n_epochs", 20, 60)
    p["batch_size"]         = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    # Darts/PL: GPU falls verfügbar
    p["pl_trainer_kwargs"]  = _dl_trainer_kwargs()
    # output_chunk_length wird im Wrapper auf H gesetzt
    return p


# ------------------------------------------------------------
# Search Space: TFT (Darts / PyTorch)
# ------------------------------------------------------------
def suggest_tft_params(trial) -> dict:
    p = {}
    p["input_chunk_length"]   = trial.suggest_int("input_chunk_length", 32, 112, step=8)
    p["hidden_size"]          = trial.suggest_categorical("hidden_size", [8, 16, 32])
    p["lstm_layers"]          = trial.suggest_int("lstm_layers", 1, 2)
    p["num_attention_heads"]  = trial.suggest_categorical("num_attention_heads", [2, 4])
    p["dropout"]              = trial.suggest_float("dropout", 0.0, 0.3)
    p["n_epochs"]             = trial.suggest_int("n_epochs", 20, 50)
    p["batch_size"]           = trial.suggest_categorical("batch_size", [64, 128, 256])
    p["pl_trainer_kwargs"]    = _dl_trainer_kwargs()
    return p


# ------------------------------------------------------------
# HPO Orchestrierung (Optuna + Rolling-Validation)
# ------------------------------------------------------------
def run_optuna_hpo(
    dataset_name: str,
    train_df: pd.DataFrame,
    max_h: int,
    train_and_score_fn: Callable[[dict, pd.DataFrame, pd.DataFrame], float],
    *,
    val_days: Optional[int] = None,
    n_folds: Optional[int] = None,
    step_days: Optional[int] = None,
    n_trials: Optional[int] = None,
    direction: Literal["minimize", "maximize"] | None = None,
    study_name: Optional[str] = None,
    sampler: Optional[object] = None,
    pruner: Optional[object] = None,
    show_progress_bar: bool = True,
    model_name: str,
    wrapper: str,
) -> dict:
    """
    Optuna-HPO mit Rolling-Validation.
    Unterstützte Modelle: LightGBM, XGBoost, N-HiTS, TFT, GNS (Global Naive Seasonal).
    Erwartung: train_and_score_fn(params, train_df_fold, val_df_fold) -> float (kleiner = besser).
    """

    # --- Defaults aus config.HPO_CFG ziehen ---
    val_days   = int(val_days   or HPO_CFG.get("val_days", 28))
    n_folds    = int(n_folds    or HPO_CFG.get("n_folds", 3))
    step_days  = int(step_days  or HPO_CFG.get("step_days", val_days))
    n_trials   = int(n_trials   or HPO_CFG.get("n_trials", 40))
    direction  =       direction or HPO_CFG.get("direction", "minimize")

    # --- Rolling-Folds bauen (nur aus train_df) ---
    train_df = sort_df(train_df)
    folds = rolling_train_val_windows(
        train_df,
        val_days=val_days,
        max_h=max_h,
        n_folds=n_folds,
        step_days=step_days,
        assume_sorted=True,
    )
    if not folds:
        raise RuntimeError(
            f"[HPO] Konnte keine Rolling-Validation-Folds erzeugen. "
            f"Prüfe train_df Umfang / val_days / n_folds / step_days."
        )

    # --- Optuna-Setup ---
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Optuna ist nicht installiert.") from e

    if sampler is None:
        sampler = TPESampler(
            seed=RANDOM_STATE,
            n_startup_trials=5,
            multivariate=True,
            constant_liar=True,
        )
    if pruner is None:
        pruner = MedianPruner(n_warmup_steps=3)

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    # --- Search space selector ---
    m = (model_name or "").lower()
    # ------------------------------------------------------------
    if m in ("lightgbm", "lgbm"):
        suggest = suggest_lgbm_params
        base_defaults = dict(LGBM_PARAMS)
        fixed = dict(
            random_state=RANDOM_STATE,
            n_jobs=LGBM_PARAMS.get('n_jobs', 6),
            verbosity=LGBM_PARAMS.get('verbosity', -1),
            force_col_wise=True,
        )
        # Übernehme GPU-spezifische Keys, falls vorhanden
        for k in ("device_type", "gpu_platform_id", "gpu_device_id"):
            if k in LGBM_PARAMS:
                fixed[k] = LGBM_PARAMS[k]
    # ------------------------------------------------------------
    elif m in ("xgboost", "xgb"):
        suggest = suggest_xgb_params
        base_defaults = dict(XGB_PARAMS)
        fixed = dict(
            random_state=RANDOM_STATE,
            n_jobs=XGB_PARAMS.get('n_jobs', 6),
            verbosity=XGB_PARAMS.get('verbosity', 0),
            tree_method=XGB_PARAMS.get('tree_method', 'hist'),
            predictor=XGB_PARAMS.get('predictor', 'auto'),
            objective='reg:squarederror',
        )
        for k in ("device", "gpu_id", "single_precision_histogram", "max_bin"):
            if k in XGB_PARAMS:
                fixed[k] = XGB_PARAMS[k]
    # ------------------------------------------------------------
    elif m == "nhits":
        suggest = suggest_nhits_params
        base_defaults = dict(NHITS_PARAMS)  # kann leer sein, wenn noch nicht in config.py
        fixed = dict(random_state=RANDOM_STATE)
    # ------------------------------------------------------------
    elif m == "tft":
        suggest = suggest_tft_params
        base_defaults = dict(TFT_PARAMS)    # kann leer sein, wenn noch nicht in config.py
        fixed = dict(random_state=RANDOM_STATE)
    # ------------------------------------------------------------
    elif m in ("gns"): # gns reicht da ich anderes nicht erlauben will
        suggest = suggest_gns_params
        base_defaults = dict(NAIVE_PARAMS)   # fix: kann leer sein
        # fixe Defaults (P=seasonal_period standardisiert z.B. 7)
        fixed = dict()  # optional leer lassen; P/K gibst du im Wrapper weiter
    else:
        raise ValueError(f"[HPO] Unbekanntes model_name: {model_name}")

    def objective(trial) -> float:
        params = suggest(trial)
        scores: List[float] = []
        for i, (tr_i, va_i) in enumerate(folds):
            score = train_and_score_fn(params, tr_i, va_i)
            scores.append(float(score))
            # Optuna-Pruning: schwache Trials früher abbrechen
            try:
                trial.report(float(score), step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except Exception:
                # falls Optuna intern nicht greift, einfach weiterlaufen
                pass
        return float(np.mean(scores))
    

    # --- Studie starten ---
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    # --- Bestes Ergebnis holen und mit Baseline mergen ---
    best_overrides = deepcopy(study.best_params)

    # Merge: Base-Defaults + HPO-Overrides + fixe Defaults
    base = dict(base_defaults)
    base.update(best_overrides)
    base.update(fixed)
    best_full = base

    # --- Optional: Automatisch speichern ---
    print(f"AUTO_SAVE_HPO = {AUTO_SAVE_HPO}")
    if AUTO_SAVE_HPO:
        try:
            save_hptuned_params(dataset_name, best_full, model_name=model_name, wrapper=wrapper)
        except Exception as e:
            print(f"[HPO] WARN: Auto-Save failed: {e}")

    # --- Hübsche Ausgabe ---
    pretty = json.dumps({model_name: {wrapper: {dataset_name: best_full}}}, indent=4, ensure_ascii=False)
    print("\n Beste HPO-Parameter (copy & paste nach config.HPTUNED_PARAMS):\n")
    print(pretty)
    print("\n Hinweis: In pipeline/run_training kannst du diese Params mit "
          "`USE_HPTUNED_PARAMS=True` + `use_hptuned_params=True` verwenden.\n")

    return best_full

# Ende hpo.py