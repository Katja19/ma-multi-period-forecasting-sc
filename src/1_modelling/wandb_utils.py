# wandb_utils.py
"""
Hilfsfunktionen für Weights & Biases:
- Run-Initialisierung mit konsistenter Ordnerstruktur
- Logging von Metriken, Feature Importance, Test-Samples
- Optional: SHAP-Summary + SHAP-Beeswarm (als PNG)
- Idempotent: pro Run wird jeder Key nur 1× geloggt
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

# wandb optional importieren (robuster in CI/Offline)
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from config import VERBOSE, AUX_KEY_COLS

__all__ = [
    "init_run",
    "log_metrics",
    "log_test_sample",
    "log_feature_importance",
    "log_shap_summary",
    "log_rmsse_per_series",
    "log_artifact_csv",
]

# =============================================================================
# Intern
# =============================================================================
def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)

def _already_logged(run, key: str) -> bool:
    """Schutz: pro Run nur 1× loggen (wir markieren im Summary)."""
    try:
        return bool(run.summary.get(key))
    except Exception:
        return False

def _mark_logged(run, key: str) -> None:
    try:
        run.summary[key] = True
    except Exception:
        pass


# =============================================================================
# Run-Init
# =============================================================================
def init_run(
    ctx: Dict,
    *,
    run_name_suffix: str,
    horizons: List[int],
    test_days: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
    model_params: Dict,
    n_series_total: int,
    n_series_train: int,
    n_series_test: int,
    project: str,
    entity: Optional[str],
    subset: bool = False,
) -> Optional[object]:
    """
    Initialisiert einen W&B-Run und legt die Config ab.

    ctx erwartet Keys: ['dir','run_name','group','tags','dataset','model','wrapper'].
    """
    if wandb is None:  # pragma: no cover
        _log("[wandb_utils] wandb not available – skipping init.")
        return None

    base_dir = Path(ctx["dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    # Run-Name: {dataset}-{model}-{wrapper}-H{max(h)}[-suffix]
    max_h = max(horizons) if horizons else 0
    suffix = f"{run_name_suffix}" if run_name_suffix and run_name_suffix != ctx.get("dataset") else ""
    run_name = ctx["run_name"] + (f"_{suffix}" if suffix else "")

    # Add timestamp only for wandb run_name/id, not elsewhere
    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #run_name_with_ts = f"{run_name}-{timestamp}"

    run = wandb.init(
        #id=run_name_with_ts, # so ist der run immer eindeutig
        project=project,
        entity=entity,
        name=run_name,
        group=ctx["group"],
        tags=ctx["tags"],
        job_type="training",
        config=dict(
            dataset=ctx["dataset"],
            subset=ctx.get("subset"),  # neu
            model=ctx["model"],
            wrapper=ctx["wrapper"], # passt so nicht ändern
            model_class=ctx.get("model_class"),
            model_wrapper=f"{ctx['model']}/{ctx['wrapper']}",
            hpTune=ctx["hpTune"],
            horizons=list(map(int, horizons)),
            test_days=int(test_days),
            n_train=int(len(train_df)),
            n_test=int(len(test_df)),
            n_features=int(len(feat_cols)),
            n_series_total=int(n_series_total),
            n_series_train=int(n_series_train),
            n_series_test=int(n_series_test),
            #lgbm_params=dict(lgbm_params),       # alte Version, ist neutral, aber besser variabe umbenennen in model_params 
            model_params=dict(model_params),      # hier ist lgbm_params neutral
        ),
        dir=str(base_dir), # einziger Pfadanker
        #dir=os.environ.get("WANDB_DIR", None),
        resume=None,
        # settings=wandb.Settings(start_method="thread"),  # deprecated / ohne Wirkung
    )
    _log(f"[wandb_utils] W&B run initialized: {run_name}")
    _log(f"[wandb_utils] local files under: {run.dir}")
    return run


# =============================================================================
# Logging
# =============================================================================    
def log_metrics(run, rmsse_total: float, rmsse_by_h: Dict[int, float], 
                *, log_table: bool = True, log_artifact: bool = True,
                artifact_name: str = "rmsse_by_horizon.csv") -> None:
    if run is None or wandb is None:
        return
    
    # Summary-Keys (Skalarwerte) – ok
    run.summary.update(
        {"rmsse_total": float(rmsse_total),
         **{f"rmsse_h{int(h)}": float(v) for h, v in rmsse_by_h.items()}}
    )

    # --- History-Table (für Charts in der UI) ---
    if log_table and rmsse_by_h and not _already_logged(run, "_rmsse_table_logged"):
        rows = [(int(h), float(v)) for h, v in sorted(rmsse_by_h.items())]
        tbl = wandb.Table(data=rows, columns=["horizon", "rmsse"])
        wandb.log({"rmsse_by_horizon": tbl})
        _mark_logged(run, "_rmsse_table_logged")

    # --- CSV-Artifact mit der per-H-Tabelle ---
    if log_artifact and rmsse_by_h and not _already_logged(run, "_rmsse_artifact_logged"):
        try:
            rows = [(int(h), float(v)) for h, v in sorted(rmsse_by_h.items())]
            df = pd.DataFrame(rows, columns=["horizon", "rmsse"])
            out = Path(run.dir) / artifact_name
            df.to_csv(out, index=False)
            art = wandb.Artifact(f"rmsse-{run.id}", type="dataset")
            art.add_file(str(out))
            run.log_artifact(art)
            _mark_logged(run, "_rmsse_artifact_logged")
        except Exception:
            pass


# =============================================================================
# Test-Sample loggen
def log_test_sample(
    run,
    series_cols: List[str],
    date_col: str,
    test_df: pd.DataFrame,
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame,
    max_rows: int = 200, # das heißt wenn ich mehr als Tage*Series habe, dann wird abgeschnitten? 
    # Antwort: ja, da ich 28 Tage habe und ca. 100 Series, dann sind es 2800 Zeilen, aber ich will nur 200 sehen
) -> None:
    """Loggt ein Sample (Keys, Datum, y, yhat) als Table – idempotent."""
    if run is None or wandb is None:
        return
    if _already_logged(run, "_test_sample_logged"):
        return
    try:
        base_cols  = list(series_cols)
        extra_cols = [c for c in AUX_KEY_COLS if c in test_df.columns and c not in base_cols]
        key_cols   = base_cols + extra_cols + [date_col]

        sample = pd.concat(
            [test_df.loc[:, key_cols], Y_true, Y_pred.add_prefix("yhat_")],
            axis=1,
            copy=False,
        ).iloc[:max_rows]
        wandb.log({"test_sample": wandb.Table(dataframe=sample)})
        _mark_logged(run, "_test_sample_logged")
    except Exception:
        pass


# =============================================================================
# Feature Importance
def log_feature_importance(run, fi_df: pd.DataFrame, top_k: int = 25,
                           *, log_scalars: bool = True, scalars_k: int = 50,
                           log_artifact: bool = True, artifact_name: str = "feature_importance.csv") -> None:
    
    if run is None or wandb is None or fi_df is None or fi_df.empty:
        return
    
    if not _already_logged(run, "_fi_logged_once"):
        fi_df = fi_df.sort_values("importance_mean", ascending=False, ignore_index=True)

        # Enrich with run context -> long table (for cross-run grouping/aggregation)
        ctx = {
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "dataset": run.config.get("dataset"),
            "model": run.config.get("model"),
            "wrapper": run.config.get("wrapper"),
            "model_wrapper": f"{run.config.get('model')}/{run.config.get('wrapper')}",
            "model_class": run.config.get("model_class"),
            "param_source": run.config.get("param_source"),
            "used_tuned_params": run.config.get("used_tuned_params"),
            "test_days": run.config.get("test_days"),
            # compact horizon string for tagging
            "horizons": "-".join(map(str, (run.config.get("horizons") or []))),
            "rmsse_total": float(run.summary.get("rmsse_total", float("nan"))),
            "rmsse_h1": float(run.summary.get("rmsse_h1", float("nan"))),
            "rmsse_h7": float(run.summary.get("rmsse_h7", float("nan"))),
            "rmsse_h14": float(run.summary.get("rmsse_h14", float("nan"))),
            "rmsse_h28": float(run.summary.get("rmsse_h28", float("nan"))),
        }
        fi_long = fi_df.copy()
        for k, v in ctx.items():
            fi_long[k] = v
        wandb.log({"fi_table_long": wandb.Table(dataframe=fi_long)})


        # Table + Bar (wie bisher)
        payload = {"feature_importance_table": wandb.Table(dataframe=fi_df)}
        try:
            k = int(min(top_k, len(fi_df)))
            if k > 0:
                top = fi_df.nlargest(k, "importance_mean")[["feature", "importance_mean"]]
                payload["feature_importance_bar"] = wandb.plot.bar(
                    wandb.Table(dataframe=top), "feature", "importance_mean",
                    title=f"Feature Importance über alle H (Top {k})",
                )
        except Exception:
            pass
        wandb.log(payload)
        

        # Summary-Skalare pro Feature (Top-K, für Aggregation über Runs)
        if log_scalars and not _already_logged(run, "_fi_scalars_logged"):
            k2 = int(min(scalars_k, len(fi_df)))
            top_feats = fi_df.nlargest(k2, "importance_mean")
            for _, r in top_feats.iterrows():
                f = str(r["feature"])
                run.summary[f"fi_mean/{f}"] = float(r["importance_mean"])
                run.summary[f"fi_std/{f}"]  = float(r.get("importance_std", 0.0))
            run.summary["fi_topk"] = list(map(str, top_feats["feature"].tolist()))
            _mark_logged(run, "_fi_scalars_logged")

        # CSV-Artifact mit kompletter FI
        if log_artifact and not _already_logged(run, "_fi_artifact_logged"):
            try:
                out = Path(run.dir) / artifact_name
                fi_df.to_csv(out, index=False)
                art = wandb.Artifact(f"fi-{run.id}", type="dataset")
                art.add_file(str(out))
                run.log_artifact(art)
                _mark_logged(run, "_fi_artifact_logged")
            except Exception:
                pass

        _mark_logged(run, "_fi_logged_once")

# =============================================================================
# SHAP
def log_shap_summary(run, shap_df: pd.DataFrame, top_k: int = 25,
                     *, log_scalars: bool = True, scalars_k: int = 50,
                     log_artifact: bool = True, artifact_name: str = "shap_summary.csv") -> None:
    if run is None or wandb is None or shap_df is None or shap_df.empty:
        return
    if not _already_logged(run, "_shap_logged_once"):
        
        # --- inside log_shap_summary(...), right after you sort shap_df ---
        shap_df = shap_df.sort_values("shap_mean_abs", ascending=False, ignore_index=True)

        ctx = {
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "dataset": run.config.get("dataset"),
            "subset": run.config.get("subset"),
            "model": run.config.get("model"),
            "model_class": run.config.get("model_class"),
            "wrapper": run.config.get("wrapper"),
            "model_wrapper": f"{run.config.get('model')}/{run.config.get('wrapper')}",
            "param_source": run.config.get("param_source"),
            "used_tuned_params": run.config.get("used_tuned_params"),
            "test_days": run.config.get("test_days"),
            "horizons": "-".join(map(str, (run.config.get("horizons") or []))),
            "rmsse_total": float(run.summary.get("rmsse_total", float("nan"))),
            "rmsse_h1": float(run.summary.get("rmsse_h1", float("nan"))),
            "rmsse_h7": float(run.summary.get("rmsse_h7", float("nan"))),
            "rmsse_h14": float(run.summary.get("rmsse_h14", float("nan"))),
            "rmsse_h28": float(run.summary.get("rmsse_h28", float("nan"))),
        }
        shap_long = shap_df.copy()
        for k, v in ctx.items():
            shap_long[k] = v
        wandb.log({"shap_table_long": wandb.Table(dataframe=shap_long)})


        # Table + Bar (wie bisher)
        payload = {"shap_table": wandb.Table(dataframe=shap_df)}
        try:
            k = int(min(top_k, len(shap_df)))
            top = shap_df.nlargest(k, "shap_mean_abs")[["feature", "shap_mean_abs"]]
            payload["shap_bar"] = wandb.plot.bar(
                wandb.Table(dataframe=top), "feature", "shap_mean_abs",
                title=f"SHAP Mean(|value|) über alle H (Top {k})",
            )
        except Exception:
            pass
        wandb.log(payload)


        # Summary-Skalare pro Feature (Top-K, für Aggregation über Runs)
        if log_scalars and not _already_logged(run, "_shap_scalars_logged"):
            k2 = int(min(scalars_k, len(shap_df)))
            top_feats = shap_df.nlargest(k2, "shap_mean_abs")
            for _, r in top_feats.iterrows():
                f = str(r["feature"])
                run.summary[f"shap_mean_abs/{f}"] = float(r["shap_mean_abs"])
            run.summary["shap_topk"] = list(map(str, top_feats["feature"].tolist()))
            _mark_logged(run, "_shap_scalars_logged")

        # CSV-Artifact mit kompletter SHAP-Summary
        if log_artifact and not _already_logged(run, "_shap_artifact_logged"):
            try:
                out = Path(run.dir) / artifact_name
                shap_df.to_csv(out, index=False)
                art = wandb.Artifact(f"shap-{run.id}", type="dataset")
                art.add_file(str(out))
                run.log_artifact(art)
                _mark_logged(run, "_shap_artifact_logged")
            except Exception:
                pass

        _mark_logged(run, "_shap_logged_once")

# =============================================================================
# Lokale Bilddatei loggen (SHAP as Beeswarm o.Ä.)
def log_image(run, key: str, image_path: str | Path) -> None:
    """
    Loggt eine lokale Bilddatei einmalig unter 'key'.
    """
    if run is None or wandb is None:
        return
    image_path = Path(image_path)
    if not image_path.exists():
        return
    mark_key = f"_{key}_logged"
    if _already_logged(run, mark_key):
        return
    wandb.log({key: wandb.Image(str(image_path))})
    _mark_logged(run, mark_key)

# --- NEU: Train-Profile (Zeit/CPU/RAM) ---------------------------------------
def log_train_profile(run, profile: Dict[str, float]) -> None:
    """
    Loggt Laufzeit- & Ressourcenprofil (Summary + Table) idempotent.
    Erwartet z.B.: {
        "train_wall_time_s": ...,
        "train_cpu_time_user_s": ...,
        "train_cpu_time_system_s": ...,
        "train_cpu_time_total_s": ...,
        "peak_rss_mb": ...,
        "rss_end_mb": ...
    }
    """
    if run is None or wandb is None or not profile:
        return
    if _already_logged(run, "_train_profile_logged"):
        return

    # Summary
    try:
        run.summary.update({k: float(v) for k, v in profile.items()})
    except Exception:
        pass

    # Table (optional – hilfreich in der UI)
    try:
        tbl = wandb.Table(data=[list(profile.values())], columns=list(profile.keys()))
        wandb.log({"train_profile": tbl})
    except Exception:
        pass

    _mark_logged(run, "_train_profile_logged")

# --- NEU: Zusatzmetriken ------------------------------------------------------
def log_regression_metrics(run, totals: Dict[str, float], by_h_table: pd.DataFrame) -> None:
    """
    - Summary: totals (mae_total, rmse_total, mape_total, r2_total)
    - Table:   by_h_table (Spalten: horizon, mae, rmse, mape, r2)
    Idempotent.
    """
    if run is None or wandb is None:
        return

    # Summary totals
    if totals:
        try:
            run.summary.update({k: float(v) for k, v in totals.items()})
        except Exception:
            pass

    # Table
    if by_h_table is not None and not by_h_table.empty and not _already_logged(run, "_extra_metrics_table_logged"):
        try:
            tbl = wandb.Table(dataframe=by_h_table)
            wandb.log({"metrics_by_horizon": tbl})
            # Optional noch kleine Lineplots:
            # for col in ("mae", "rmse", "mape", "r2"):
            #     try:
            #         plot = wandb.plot.line(tbl, "horizon", col, title=f"{col.upper()} by horizon")
            #         wandb.log({f"{col}_line": plot})
            #     except Exception:
            #         pass
            _mark_logged(run, "_extra_metrics_table_logged")
        except Exception:
            pass

# =============================================================================
# Pro-Series RMSSE (Top-N + Histogramm)
# # =============================================================================
def log_rmsse_per_series(
    run,
    df: pd.DataFrame,
    top_n: int = 100,
    *,
    id_col: str = "id",
    item_col: str = "item_id",
    store_col: str = "store_id",
) -> None:
    """
    Loggt RMSSE-Tabellen:
      - Top-N pro Serie (id)
      - optional: Top-N pro item_id (mean-RMSSE)
      - optional: Top-N pro store_id (mean-RMSSE)
    und speichert je Sicht ein CSV-Artifact (vollständige Tabelle).
    Idempotent (loggt genau einmal pro Run).
    """
    if run is None or wandb is None or df is None or df.empty:
        return
    if _already_logged(run, "_rmsse_series_logged"):
        return

    # Metadaten für die Artefakte (hilfreich für spätere Suche/Filter)
    meta = {
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "dataset": run.config.get("dataset"),
        "subset": run.config.get("subset"),
        "model": run.config.get("model"),
        "wrapper": run.config.get("wrapper"),
        "model_class": run.config.get("model_class") or run.config.get("family"),
        "horizons": run.config.get("horizons"),
        "test_days": run.config.get("test_days"),
    }

    def _emit(view_name: str, df_full: pd.DataFrame, table_key: str) -> None:
        if df_full is None or df_full.empty:
            return
        # Top-N Tabelle in die UI
        top = df_full.head(int(top_n))
        run.log({table_key: wandb.Table(dataframe=top)})
        # CSV + Artifact
        try:
            csv_path = Path(run.dir) / f"{view_name}.csv"
            df_full.to_csv(csv_path, index=False)
            art = wandb.Artifact(
                name=view_name,               # W&B versioniert automatisch je Run
                type="dataset",
                description=f"{view_name.replace('_',' ')} (full table)",
                metadata=meta,
            )
            art.add_file(str(csv_path))
            run.log_artifact(art)
        except Exception:
            pass

    # --- by id (always when present) ---
    if id_col in df.columns:
        by_id_full = (
            df.loc[:, [id_col, "rmsse"]]
              .sort_values("rmsse", ascending=False)
              .reset_index(drop=True)
        )
        _emit("rmsse_by_id", by_id_full, "rmsse_per_series_top")

    # --- by item (optional) ---
    if item_col in df.columns:
        by_item_full = (
            df.groupby(item_col, as_index=False)["rmsse"]
              .mean()
              .sort_values("rmsse", ascending=False, ignore_index=True)
        )
        _emit("rmsse_by_item", by_item_full, "rmsse_by_item_top")

    # --- by store (optional) ---
    if store_col in df.columns:
        by_store_full = (
            df.groupby(store_col, as_index=False)["rmsse"]
              .mean()
              .sort_values("rmsse", ascending=False, ignore_index=True)
        )
        _emit("rmsse_by_store", by_store_full, "rmsse_by_store_top")

    _mark_logged(run, "_rmsse_series_logged")



# =============================================================================
# CSV-Artifact loggen
# =============================================================================
def log_artifact_csv(run, df: pd.DataFrame, name: str, description: str = "") -> None:
    """
    Schreibt df als CSV in den Run-Ordner und logged es als W&B-Artifact (type=dataset).
    """
    if run is None or wandb is None or df is None or df.empty:
        return
    try:
        out_dir = Path(run.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        art = wandb.Artifact(name=name, type="dataset", description=description or name)
        art.add_file(str(csv_path), name=f"{name}.csv")
        run.log_artifact(art)
    except Exception:
        pass

# =============================================================================
# Run-Finish
# =============================================================================
def finish_run(run, silent: bool = True) -> None:
    """Schließt den W&B-Run (falls vorhanden)."""
    if run is None or wandb is None:
        return
    try:
        run.finish(exit_code=0)
        _log("[wandb_utils] W&B run finished.")
    except Exception:
        pass
    
# Ende wandb_utils.py