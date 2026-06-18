# results_logger.py
from __future__ import annotations

"""
Kompakter Run-Logger:
- Hängt einen Run an eine Master-CSV an (outputs/evaluation/runs_summary.csv)
- Baut einfache Aggregationen (nach Modell/Wrapper/Dataset/Klasse)
- Verändert keine Trainingslogik; nur Logging/CSV-Ausgabe
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd

# ---------------------------------------------------------------------
# Modellklassifizierung (Primär: aus config; Fallback lokal)
# ---------------------------------------------------------------------
try:
    from config import model_class_of, WANDB_DIR_BASE
except Exception:
    def model_class_of(name: str) -> str:
        m = (name or "").lower()
        if m in ("lgbm", "lightgbm", "xgboost", "xgb"):
            return "tree_ensemble"
        if m in ("nhits", "n-hits", "tft", "temporal_fusion_transformer"):
            return "deep_learning"
        if m in ("gns", "naive", "seasonal_naive"):
            return "naive"
        return "other"
    WANDB_DIR_BASE = Path.cwd() / "outputs" / "wandb"  # konsistenter Default

# ---------------------------------------------------------------------
# kleine Utilities
# ---------------------------------------------------------------------
def _short_model_name(m: str) -> str:
    m = (m or "").lower()
    return m.replace("lightgbm", "lgbm").replace("xgboost", "xgb")

def _wrapper_display(model: str, wrapper: str) -> str:
    # Für DL-Modelle „native“ anzeigen (Konsistenz mit DL-Wrappern)
    if (model or "").lower() in ("nhits", "tft", "temporal_fusion_transformer", "n-hits"):
        return "native"
    return (wrapper or "").lower()

def _param_label(model: str, result: Dict[str, Any]) -> str:
    """
    'tuned' | 'explicit' | 'default' (+ bei GNS ggf. 'K=...').
    """
    used_tuned = bool(result.get("used_tuned_params", False))
    src = (result.get("param_source") or "").lower()
    label = "tuned" if used_tuned or src in ("fresh_hpo", "saved") else ("explicit" if src == "explicit" else "default")
    if (model or "").lower() == "gns":
        try:
            k = result.get("params", {}).get("K", None)
            if k is not None:
                label = f"{label}; K={int(k)}"
        except Exception:
            pass
    return label

def _run_name_fallback(cfg, result) -> str:
    base = f"{_short_model_name(cfg.model)}-{'hpTuned' if getattr(cfg, 'use_hptuned_params', False) else 'basic'}-{cfg.wrapper}-{cfg.dataset}"
    return f"{base}_{getattr(cfg, 'run_suffix', '')}".rstrip("_")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _outdir() -> Path:
    # gleiche Root-Logik wie in config: outputs/evaluation neben outputs/wandb
    return WANDB_DIR_BASE.parent / "evaluation"

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _collect_extra_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nimmt zusätzliche Kennzahlen (falls vorhanden) dynamisch mit auf.
    z.B. rmsse_* / rmse / mae / mape / smape …
    """
    extra = {}
    if not isinstance(result, dict):
        return extra
    for k, v in result.items():
        kl = str(k).lower()
        if any(kl.startswith(pfx) for pfx in ("rmsse", "rmse", "mae", "mape", "smape")):
            try:
                extra[k] = float(v)
            except Exception:
                pass
    return extra

# =============================================================================
# Hauptfunktion
# =============================================================================
def append_run_summary(cfg, result: Dict[str, Any]) -> Dict[str, Path]:
    """
    Hängt einen Run an runs_summary.csv an und erzeugt Aggregations-CSV.
    Erwartet:
      - cfg: ExperimentConfig (oder Objekt mit gleichnamigen Feldern)
      - result: Rückgabe von pipeline.run_training(...)

    Rückgabe: Pfade der erzeugten Dateien.
    """
    # Trainings mit HPO oder Subset nicht loggen (wie bisher)
    if getattr(cfg, "subset", False) or getattr(cfg, "run_hpo", False):
        return {}

    outdir = _outdir()
    _ensure_dir(outdir)

    # Kernfelder
    model_class = model_class_of(cfg.model)
    model_short = _short_model_name(cfg.model)
    wrapper_disp = _wrapper_display(cfg.model, cfg.wrapper)
    param_txt = _param_label(cfg.model, result)
    try:
        rmsse_total = float(result.get("rmsse_total", float("nan")))
    except Exception:
        rmsse_total = float("nan")

    # W&B Infos (optional)
    run_ref = result.get("wandb_run", None)
    run_name = getattr(run_ref, "name", None) or _run_name_fallback(cfg, result)
    run_url  = getattr(run_ref, "url", None) or ""

    # Grundzeile
    row = {
        "Model_Klasse": model_class,
        "Model": model_short,
        "Wrapper": wrapper_disp,
        "Parameter": param_txt,
        "Datensatz": getattr(cfg, "dataset", ""),
        "RMSSE_total": rmsse_total,
        "Run_Name": run_name,
        "Param_Source": result.get("param_source"),
        "Used_Tuned_Params": bool(result.get("used_tuned_params", False)),
        "Run_Suffix": getattr(cfg, "run_suffix", ""),
        "Timestamp": _utc_now_iso(),
        "WandB_URL": run_url,
    }
    # optionale Zusatzmetriken anhängen (ohne bestehendes Schema zu brechen)
    row.update(_collect_extra_metrics(result))
    df_row = pd.DataFrame([row])

    # 1) Master-CSV anhängen
    master_csv = outdir / "runs_summary.csv"
    header = not master_csv.exists() or master_csv.stat().st_size == 0
    df_row.to_csv(master_csv, index=False, mode="a", header=header)

    # 2) Aggregationen
    _build_aggregates(master_csv, outdir)

    return {
        "summary_csv": master_csv,
        "agg_by_model_wrapper_dataset": outdir / "agg_by_model_wrapper_dataset.csv",
        "agg_by_model_wrapper": outdir / "agg_by_model_wrapper.csv",
        "agg_by_class": outdir / "agg_by_class.csv",
        "best_per_dataset": outdir / "best_per_dataset.csv",
    }

# ---------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------
def _build_aggregates(master_csv: Path, outdir: Path) -> None:
    if not master_csv.exists():
        return
    try:
        df = pd.read_csv(master_csv)
    except Exception:
        return

    # defensive typing
    if "RMSSE_total" in df.columns:
        df["RMSSE_total"] = pd.to_numeric(df["RMSSE_total"], errors="coerce")

    # 2a) Mittelwert je (Model, Wrapper, Datensatz)
    cols_a = {"Model","Wrapper","Datensatz","RMSSE_total"}
    if cols_a.issubset(df.columns):
        gb = df.groupby(["Model","Wrapper","Datensatz"], as_index=False)["RMSSE_total"].mean()
        gb.to_csv(outdir / "agg_by_model_wrapper_dataset.csv", index=False)

    # 2b) Mittelwert je (Model, Wrapper) über alle Datasets
    cols_b = {"Model","Wrapper","RMSSE_total"}
    if cols_b.issubset(df.columns):
        gb2 = df.groupby(["Model","Wrapper"], as_index=False)["RMSSE_total"].mean()
        gb2.to_csv(outdir / "agg_by_model_wrapper.csv", index=False)

    # 2c) Mittelwert je Modellklasse
    cols_c = {"Model_Klasse","RMSSE_total"}
    if cols_c.issubset(df.columns):
        gb3 = df.groupby(["Model_Klasse"], as_index=False)["RMSSE_total"].mean()
        gb3.to_csv(outdir / "agg_by_class.csv", index=False)

    # 2d) Bestes Modell je Datensatz (min RMSSE_total)
    cols_d = {"Datensatz","RMSSE_total"}
    if cols_d.issubset(df.columns):
        best = (df.sort_values(["Datensatz","RMSSE_total"], ascending=[True, True])
                  .groupby("Datensatz", as_index=False).head(1))
        best.to_csv(outdir / "best_per_dataset.csv", index=False)
        
# ----------------------------------------------------------------------
# Alternative API: Payload bauen + schreiben (für interaktive Nutzung)
# ----------------------------------------------------------------------
def build_log_payload(cfg, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baut ein Log-Payload (eine einzelne Zeile als Dict) + Zielpfade,
    schreibt aber NICHT auf die Platte. Für interaktive Freigabe im Notebook.
    """
    model_class = model_class_of(cfg.model)
    model_short = _short_model_name(cfg.model)
    wrapper_disp = _wrapper_display(cfg.model, cfg.wrapper)
    param_txt = _param_label(cfg.model, result)
    try:
        rmsse_total = float(result.get("rmsse_total", float("nan")))
    except Exception:
        rmsse_total = float("nan")

    run_ref = result.get("wandb_run", None)
    run_name = getattr(run_ref, "name", None) or _run_name_fallback(cfg, result)
    run_url  = getattr(run_ref, "url", None) or ""

    row = {
        "Model_Klasse": model_class,
        "Model": model_short,
        "Wrapper": wrapper_disp,
        "Parameter": param_txt,
        "Datensatz": getattr(cfg, "dataset", ""),
        "RMSSE_total": rmsse_total,
        "Run_Name": run_name,
        "Param_Source": result.get("param_source"),
        "Used_Tuned_Params": bool(result.get("used_tuned_params", False)),
        "Run_Suffix": getattr(cfg, "run_suffix", ""),
        "Timestamp": _utc_now_iso(),
        "WandB_URL": run_url,
    }
    row.update(_collect_extra_metrics(result))

    outdir = _outdir()
    master_csv = outdir / "runs_summary.csv"
    return {
        "row": row,
        "master_csv": str(master_csv),
    }

# ----------------------------------------------------------------------
# Payload schreiben + Aggregationen aktualisieren
def append_log_payload(payload: Dict[str, Any]) -> Dict[str, Path]:
    """
    Nimmt ein via build_log_payload erzeugtes Payload und schreibt es
    in runs_summary.csv + aktualisiert die Aggregations.
    """
    p_master = Path(payload.get("master_csv", _outdir() / "runs_summary.csv"))
    outdir = p_master.parent
    _ensure_dir(outdir)

    df_row = pd.DataFrame([payload["row"]])
    header = not p_master.exists() or p_master.stat().st_size == 0
    df_row.to_csv(p_master, index=False, mode="a", header=header)

    _build_aggregates(p_master, outdir)

    return {
        "summary_csv": p_master,
        "agg_by_model_wrapper_dataset": outdir / "agg_by_model_wrapper_dataset.csv",
        "agg_by_model_wrapper": outdir / "agg_by_model_wrapper.csv",
        "agg_by_class": outdir / "agg_by_class.csv",
        "best_per_dataset": outdir / "best_per_dataset.csv",
    }

# Ende results_logger.py =============================================
