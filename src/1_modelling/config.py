# config.py
from __future__ import annotations

# ======================================================================
# Standardbibliothek
# ======================================================================
import os
import json
import tempfile
from json import JSONDecodeError
from pathlib import Path
from typing import Sequence
from datetime import datetime

# ======================================================================
# Projekt-Imports
# ======================================================================
from datasets import get_spec, DatasetSpec

# ======================================================================
# Logging / Verbosität
# ======================================================================
VERBOSE: bool = True
LOG_PREFIX: str = "CFG"

# ======================================================================
# Reproduzierbarkeit
# ======================================================================
RANDOM_STATE: int = 42

def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """
    Setzt Seeds für Python, NumPy und (falls vorhanden) PyTorch.
    """
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # LightGBM nutzt den übergebenen random_state; kein globaler Seed nötig.

# ======================================================================
# Modellfamilien-Erkennung (für konsistente W&B-Tags etc.)
# ======================================================================
def model_class_of(name: str) -> str:
    m = (name or "").lower()
    if m in ("lgbm", "lightgbm", "xgboost", "xgb"):
        return "tree_ensemble"
    if m in ("n-hits", "n_hits", "nhits"):
        return "deep_learning"
    if m in ("tft", "temporal_fusion_transformer"):
        return "deep_learning"
    if m in ("seasonal_naive", "naive", "seasonalnaive", "gns"):
        return "naive"
    return "other"

# ======================================================================
# Dataset & Spalten
# ======================================================================
# Aktives Dataset ("bakery" | "m5")
DATASET_NAME: str = "bakery"
DATASET_SPEC: DatasetSpec = get_spec(DATASET_NAME)

# Einheitliche Spaltennamen aus dem aktiven Dataset-Spec
DATE_COL: str = DATASET_SPEC.date_col            # z. B. "date"
TARGET_COL: str = DATASET_SPEC.target_col        # z. B. "demand"
SERIES_COLS: Sequence[str] = DATASET_SPEC.series_cols  # z. B. ("id",)

# Optionale kategorische Spalten (falls vorhanden)
CATEGORICAL_COLS: tuple[str, ...] = ("weekday", "month", "year", "week")

# ======================================================================
# Forecasting-Defaults
# ======================================================================
SEASONAL_PERIOD: int = 7
EVAL_HORIZONS_DEFAULT: tuple[int, ...] = tuple(range(1, 29))
N_TEST_ORIGINS_DEFAULT: int = 56

# ======================================================================
# Modell- & Trainings-Defaults
# ======================================================================
MODEL_NAME: str = "lightgbm"  # "lightgbm" | "xgboost" | "n-hits" | "tft" | "gsn"

# ---- GlobalNaiveSeasonal (flexibel) ----
NAIVE_PARAMS: dict = dict(
    seasonal_period=7,
    K=1,  # Standard; im Notebook beliebig überschreiben (z. B. 2, 7, 28, ...)
)

# ---- XGBoost FAST Defaults (GPU-freundlich) ----
XGB_PARAMS: dict = dict(
    n_jobs=6,
    n_estimators=500,        # 1000 -> 500  (≈ 2x schneller)
    learning_rate=0.1,     # 0.045 -> 0.1 (weniger Bäume nötig)
    max_depth=6,            # 8 -> 6        (schneller, robuster)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,         # 1.5 -> 1.0
    reg_alpha=0.0,
    min_child_weight=3.0,   # 2.0 -> 3.0    (glättet, beschleunigt Splits)
    max_bin=256,            # 512 -> 256    (schnelleres Histogramm)
    single_precision_histogram=True,  # (XGB >=2, GPU) schneller/kleiner
    random_state=RANDOM_STATE,
    verbosity=0,            # 0 bei xgb = -1 bei lgbm
    tree_method="hist",     # GPU wird in der Pipeline injiziert
    objective="reg:squarederror",
)

# ---- LightGBM FAST Defaults (GPU-freundlich) ----
LGBM_PARAMS: dict = dict(
    n_jobs=6,
    n_estimators=700,       # 1500 -> 700  (≈ 2x schneller)
    learning_rate=0.10,     # 0.04 -> 0.10 (weniger Bäume nötig)
    max_depth=-1,
    num_leaves=95,          # 192 -> 95   (kompaktere Bäume)
    max_bin=191,            # 255 -> 191  (schnelleres Histogramm)
    min_child_samples=80,   # 60 -> 80    (stabiler, weniger Overfit)
    subsample=0.8,
    colsample_bytree=0.8,
    bagging_freq=1,
    force_col_wise=True,
    reg_lambda=1.0,         # 1.5 -> 1.0
    reg_alpha=0.0,
    random_state=RANDOM_STATE,
    verbosity=-1,
    FORCE_TREES_ON_GPU=True,  # zentraler Schalter (kann in der Pipeline überschrieben werden)
    PREFERRED_GPU_ID=0,      # zentrale GPU-ID (kann in der Pipeline überschrieben werden)
)
# ======================================================================
# Einheitliche GPU-Nutzung für Bäume (Vergleichbarkeit zu DL-Modellen)
# ======================================================================
# Immer GPU 0 (meine RTX 3050) verwenden
PREFERRED_GPU_ID = 0

def preferred_gpu_id() -> int:
    return int(PREFERRED_GPU_ID)

def force_cuda_device_env(gpu_id: int | None = None) -> None:
    """MUSS aufgerufen werden, bevor xgboost/torch importiert werden."""
    gid = PREFERRED_GPU_ID if gpu_id is None else int(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

# Trees standardmäßig auf GPU (kannst du auf False setzen, wenn nötig)
FORCE_TREES_ON_GPU = True

# LightGBM: explizit NVIDIA-OpenCL wählen (bei Intel iGPU sonst oft platform_id=0)
LGBM_PARAMS.update({
    "device_type": "gpu",
    "gpu_platform_id": 1,   # bei vielen Laptops: 0=Intel, 1=NVIDIA
    "gpu_device_id": 0,     # erste NVIDIA-Karte
}) 
# == =====================================================================




# ---- Optionale GPU-Flags über Umgebungsvariablen ----
# XGBoost:   set XGB_USE_GPU=1  (Windows)  /  export XGB_USE_GPU=1  (Linux/Mac)
if os.getenv("XGB_USE_GPU", "0") == "1":
    XGB_PARAMS.update({"device": "cuda"})

# LightGBM:  set LGBM_USE_GPU=1 (Windows)  /  export LGBM_USE_GPU=1 (Linux/Mac)
# (erfordert eine LightGBM-Build mit GPU/OpenCL; CPU-Fallback in den Modellen vorhanden)
if os.getenv("LGBM_USE_GPU", "0") == "1":
    LGBM_PARAMS.update(dict(device_type="gpu", gpu_platform_id=0, gpu_device_id=0))

# SHAP-Logging (kann Laufzeit erhöhen)
LOG_SHAP: bool = True

# ======================================================================
# Weights & Biases (W&B)
# ======================================================================
AUX_KEY_COLS: tuple[str, ...] = ("item_id", "store_id")  # optionale Schlüsselspalten für Logging

ROOT: Path = Path.cwd().parents[1] if len(Path.cwd().parents) >= 2 else Path.cwd()
WANDB_DIR_BASE: Path = ROOT / "outputs" / "wandb"

USE_WANDB: bool = True
WANDB_PROJECT: str = "multi-period-forecasting"
WANDB_ENTITY: str | None = None  # z. B. "katja-gegg" oder None

def wandb_run_context(
    model_name: str,
    wrapper: str,
    dataset_name: str,
    subset: bool,
    *,
    eval_horizons: list[int] | None = None,
    n_test_origins: int | None = None,
    use_hptuned_params: bool = False,
    run_hpo: bool = False,
    used_tuned_params: bool | None = None,
    params_source: str | None = None,   # "explicit" | "fresh_hpo" | "saved" | "default"
    model_params: dict | None = None,   # für GNS K-Wert im Namen
) -> dict:
    """
    Liefert konsistente W&B-Metadaten (Name/Group/Tags) und Zielordner.
    """
    subdir = f"{model_name}-{wrapper}-{dataset_name}"
    dirpath = WANDB_DIR_BASE / subdir

    hp_or_basic = "hpTuned" if use_hptuned_params else "basic"
    short_model_name = (
        model_name.lower()
        .replace("lightgbm", "lgbm")
        .replace("xgboost", "xgb")
    )

    # Spezieller Run-Name für GNS (K im Namen)
    if short_model_name in ("gns", "seasonal_naive", "seasonalnaive", "naive"):
        K_val = None
        try:
            tuned = (
                HPTUNED_PARAMS
                .get("gns", {})
                .get(wrapper.lower(), {})
                .get(dataset_name, {})
            )
            K_val = tuned.get("K")
        except Exception:
            pass
        if K_val is None:
            K_val = (model_params or {}).get("K")
        if K_val is None:
            K_val = NAIVE_PARAMS.get("K")
        run_name = f"gns-K{int(K_val)}-{hp_or_basic}-{wrapper}-{dataset_name}"
    else:
        run_name = f"{short_model_name}-{hp_or_basic}-{wrapper}-{dataset_name}"

    group = f"{model_name}"
    model_class = model_class_of(model_name)

    tags = [
        f"dataset:{dataset_name}",
        f"subset:{subset}",
        f"model:{model_name}",
        f"wrapper:{wrapper}",
        f"model_class:{model_class}",
        f"model_wrapper:{model_name}/{wrapper}",
        f"params_scope:{model_name}/{wrapper}/{dataset_name}",
        f"hpTuned:{'yes' if use_hptuned_params else 'no'}",
        "global:true",
        "target:univariate",
        "features:multivariate",
        "split:origin",
        f"hpo_run:{'yes' if run_hpo else 'no'}",
        f"hpo_used:{'yes' if used_tuned_params else 'no'}",
        f"hpo_metric:{HPO_CFG.get('metric','')}",
    ]

    if params_source:
        tags.append(f"params_source:{params_source}")
    if n_test_origins is not None:
        tags.append(f"test_origins:{int(n_test_origins)}")
    if eval_horizons:
        ev = sorted(map(int, eval_horizons))
        tags.append("eval:" + ("-".join(map(str, ev)) if len(ev) <= 6 else f"{ev[0]}..{ev[-1]}"))
        tags.append(f"max_h:{max(ev)}")

    if VERBOSE:
        for t in tags:
            print(f"[{LOG_PREFIX}:wandb_run_context] tag: {t}")

    return {
        "run_name": run_name,
        "group": group,
        "model_class": model_class,
        "subset": subset,
        "model_wrapper": f"{model_name}/{wrapper}",
        "tags": tags,
        "dir": dirpath,
        # Keys, die wandb_utils.init_run() erwartet:
        "dataset": dataset_name,
        "model": model_name,
        "wrapper": wrapper,
        "n_series": None,
        "hpTune": "yes" if use_hptuned_params else "no",
    }

# ======================================================================
# HPO / Tuning
# ======================================================================
# Master-Switch: Nutzung getunter Parameter (Notebook-Flag kann pro Run ausschalten)
USE_HPTUNED_PARAMS: bool = True

# Persistenter Speicher: model -> wrapper -> dataset -> params
HPTUNED_PARAMS: dict[str, dict[str, dict[str, dict]]] = {}

# Optional: MIMO-Tunings zusätzlich auf DIRECT spiegeln (und umgekehrt, falls gewünscht)
AUTO_MIRROR_HPO: bool = True
HPO_MIRROR_MAP: dict[tuple[str, str], list[str]] = {
    ("lightgbm", "mimo"): ["direct"],
    ("xgboost",  "mimo"): ["direct"],
}

# Optuna-Defaults
HPO_CFG: dict = dict(
    val_days=14,
    n_folds=3,
    step_days=14,
    n_trials=16,
    direction="minimize",
    metric="rmsse",
)
# Welche Horizonte HPO optimiert (kann kleiner sein als Eval-Horizonte)
HPO_CFG["hpo_horizons"] = (1, 7, 14, 28)
HPO_CFG["early_stopping_rounds"] = 50

def get_params_for_dataset(
    model_name: str,
    wrapper: str,
    dataset_name: str,
    *,
    allow_tuned: bool,
    fallback: dict | None = None,
) -> tuple[dict, bool]:
    """
    Liefert (params, used_tuned_flag) aus HPTUNED_PARAMS oder fällt auf Defaults zurück.
    """
    base = fallback or (XGB_PARAMS if model_name.lower() == "xgboost" else LGBM_PARAMS)
    if allow_tuned:
        tuned = (
            HPTUNED_PARAMS
            .get(model_name.lower(), {})
            .get(wrapper.lower(), {})
            .get(dataset_name, None)
        )
        if tuned:
            return dict(tuned), True
    return dict(base), False

# ======================================================================
# HPO-Persistenz (JSON-Datei)
# ======================================================================
AUTO_SAVE_HPO: bool = True
WANDB_DIR_BASE = WANDB_DIR_BASE  # nur für Lesbarkeit (gleicher Pfad wie oben)
HPTUNED_PARAMS_FILE: Path = WANDB_DIR_BASE.parent / "hptuned_params.json"

def load_hptuned_params_from_file() -> None:
    """
    Lädt persistente Tunings tief-verschachtelt und merged in HPTUNED_PARAMS.
    """
    global HPTUNED_PARAMS
    try:
        if not HPTUNED_PARAMS_FILE.exists():
            return
        with open(HPTUNED_PARAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for model_name, wmap in data.items():
            if not isinstance(wmap, dict):
                continue
            HPTUNED_PARAMS.setdefault(model_name, {})
            for wrapper, dmap in wmap.items():
                if not isinstance(dmap, dict):
                    continue
                HPTUNED_PARAMS[model_name].setdefault(wrapper, {})
                HPTUNED_PARAMS[model_name][wrapper].update(dmap)
        if VERBOSE:
            print(f"[CFG] Loaded tuned params from {HPTUNED_PARAMS_FILE}")
    except (JSONDecodeError, OSError) as e:
        if VERBOSE:
            print(f"[CFG] WARN: could not load {HPTUNED_PARAMS_FILE}: {e}")

def save_hptuned_params(
    dataset_name: str,
    params: dict,
    *,
    model_name: str,
    wrapper: str,
) -> None:
    """
    Speichert/merged Tunings unter model/wrapper/dataset; spiegelt optional MIMO→DIRECT.
    """
    model_key = model_name.lower()
    wrapper_key = wrapper.lower()

    # aktuelle Datei robust einlesen
    current: dict = {}
    try:
        if HPTUNED_PARAMS_FILE.exists():
            with open(HPTUNED_PARAMS_FILE, "r", encoding="utf-8") as f:
                current = json.load(f) or {}
        if not isinstance(current, dict):
            current = {}
    except (JSONDecodeError, OSError):
        current = {}

    def _set_entry(store: dict, m: str, w: str, ds: str, p: dict) -> None:
        store.setdefault(m, {}).setdefault(w, {})[ds] = dict(p)

    # 1) Primärpfad
    _set_entry(current, model_key, wrapper_key, dataset_name, params)
    _set_entry(HPTUNED_PARAMS, model_key, wrapper_key, dataset_name, params)

    # 2) Mirrors (Option C)
    if AUTO_MIRROR_HPO:
        for (m, src), dst_list in HPO_MIRROR_MAP.items():
            if m == model_key and src == wrapper_key:
                for dst in dst_list:
                    _set_entry(current,        model_key, dst, dataset_name, params)
                    _set_entry(HPTUNED_PARAMS, model_key, dst, dataset_name, params)

    # 3) Atomar speichern
    try:
        HPTUNED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(HPTUNED_PARAMS_FILE.parent), encoding="utf-8") as tmp:
            json.dump(current, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, HPTUNED_PARAMS_FILE)
        if VERBOSE:
            mirrors = HPO_MIRROR_MAP.get((model_key, wrapper_key), []) if AUTO_MIRROR_HPO else []
            print(
                f"[CFG] Saved tuned params for '{model_key}/{wrapper_key}/{dataset_name}'"
                + (f" (mirrored → {mirrors})" if mirrors else "")
                + f" → {HPTUNED_PARAMS_FILE}"
            )
    except OSError as e:
        if VERBOSE:
            print(f"[CFG] WARN: could not write {HPTUNED_PARAMS_FILE}: {e}")
        try:
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# Beim Import einmalig laden
load_hptuned_params_from_file()

# ======================================================================
# DL-Validation Summaries (kein HPO!)
# ======================================================================
DL_VAL_SUMMARY_FILE: Path = WANDB_DIR_BASE.parent / "dl_val_summaries.json"

def _json_sanitize(obj):
    """Macht Dicts JSON-sicher (nur einfache Typen rekursiv behalten)."""
    import numpy as _np
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            vv = _json_sanitize(v)
            if isinstance(vv, (dict, list)) or vv is None or isinstance(vv, (bool, int, float, str)):
                out[ks] = vv
        return out
    # alles andere verwerfen
    return None

def save_dl_val_summary(
    dataset_name: str,
    *,
    model_name: str,
    wrapper: str = "native",
    used_params: dict,
    val_days: int,
    train_meta: dict | None = None,
) -> None:
    """
    Speichert eine *Validation*-Zusammenfassung für DL-Modelle (kein HPO!).
    Struktur: model -> wrapper -> dataset -> [ {timestamp, params, meta}, ... ]
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "used_params": _json_sanitize(dict(used_params or {})),
        "val_days": int(val_days),
        "meta": _json_sanitize(dict(train_meta or {})),
    }

    # Datei robust laden/mergen
    data: dict = {}
    try:
        if DL_VAL_SUMMARY_FILE.exists():
            with open(DL_VAL_SUMMARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        if not isinstance(data, dict):
            data = {}
    except (JSONDecodeError, OSError):
        data = {}

    data.setdefault(model_name.lower(), {}).setdefault(wrapper.lower(), {}).setdefault(dataset_name, []).append(record)

    # atomar speichern
    try:
        DL_VAL_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(DL_VAL_SUMMARY_FILE.parent), encoding="utf-8") as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, DL_VAL_SUMMARY_FILE)
        if VERBOSE:
            print(f"[CFG] DL val summary appended → {DL_VAL_SUMMARY_FILE}")
    except OSError as e:
        if VERBOSE:
            print(f"[CFG] WARN: could not write {DL_VAL_SUMMARY_FILE}: {e}")
        try:
            if 'tmp_path' in locals():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# ======================================================================
# Einheitliche GPU-Nutzung für Bäume (Vergleichbarkeit zu DL-Modellen)
# ======================================================================
FORCE_TREES_ON_GPU: bool = True  # zentraler Schalter

def preferred_gpu_id() -> int:
    """
    Wählt denselben GPU-Index wie PyTorch (falls verfügbar), sonst heuristisch 0.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    env = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        try:
            return int(env.split(",")[0])
        except Exception:
            return 0
    return 0

# Ende config.py