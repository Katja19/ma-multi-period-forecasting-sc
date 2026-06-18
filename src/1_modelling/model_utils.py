# model_utils.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Dict
import numpy as np
import pandas as pd


__all__ = [
    # Logging helper
    "clog",
    # Estimator access
    "get_estimators",
    # Feature Importance (über alle H)
    "feature_importance_df",
    # SHAP (über alle H)
    "shap_values_aggregate",   # -> (X_rep, shap_vals_agg)
    "shap_summary_df",         # Mean(|SHAP|) über alle H
    "shap_beeswarm_figure",    # Beeswarm aus aggregierten SHAP-Werten
    # Metriken
    "regression_metrics_total",
    "regression_metrics_table",
    "rmsse_per_series_df",
    "detect_compute_backend",
]

# --- NEU: Zusatzmetriken -----------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    denom = np.where(np.abs(t) < eps, np.nan, np.abs(t))
    return float(np.nanmean(np.abs(p - t) / denom))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(t) & np.isfinite(p)
    if not np.any(mask):
        return np.nan
    t = t[mask]; p = p[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - np.nanmean(t))**2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def regression_metrics_total(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> Dict[str, float]:
    """MAE, RMSE, MAPE, R2 über alle Punkte/Horizonte (vektorisiert)."""
    t = Y_true.to_numpy(dtype=float)
    p = Y_pred.to_numpy(dtype=float)
    diff = p - t
    mae  = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff**2)))
    mape = _safe_mape(t, p)
    r2   = _r2(t, p)
    return {"mae_total": mae, "rmse_total": rmse, "mape_total": mape, "r2_total": r2}

def regression_metrics_table(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Per-Horizont Tabelle: ['horizon','mae','rmse','mape','r2'], aufsteigend nach horizon.
    """
    rows = []
    # Horizonte robust aus den Spaltennamen lesen
    hs, cols = [], []
    for c in Y_true.columns:
        try:
            h = int(str(c).split("+", 1)[1]); hs.append(h); cols.append(c)
        except Exception:
            continue
    order = [c for _, c in sorted(zip(hs, cols), key=lambda x: x[0])]
    hs_sorted = sorted(hs)

    for h, c in zip(hs_sorted, order):
        t = Y_true[c].to_numpy(dtype=float)
        p = Y_pred[c].to_numpy(dtype=float)
        diff = p - t
        mae  = float(np.nanmean(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        mape = _safe_mape(t, p)
        r2   = _r2(t, p)
        rows.append((h, mae, rmse, mape, r2))

    df = pd.DataFrame(rows, columns=["horizon", "mae", "rmse", "mape", "r2"])
    return df.sort_values("horizon", ignore_index=True)


# =============================================================================
# Light logging helper
# =============================================================================
def clog(scope: str, msg: str) -> None:
    """Kleiner Logger, schaltbar via config.VERBOSE."""
    try:
        from config import VERBOSE, LOG_PREFIX
    except Exception:
        VERBOSE, LOG_PREFIX = True, "LOG"
    if VERBOSE:
        print(f"[{LOG_PREFIX}:{scope}] {msg}")


# =============================================================================
# Estimator-Ermittlung (Direct & MIMO)
# =============================================================================
def get_estimators(model) -> List:
    """
    Vereinheitlichte Schnittstelle für Direct (models_) & MIMO (model_.estimators_).
    Gibt die Basismodelle in stabiler Reihenfolge zurück.
    - Direct: dict {h: model}  -> Reihenfolge anhand model.horizons (falls vorhanden)
    - MIMO:   MultiOutputRegressor.estimators_ (Liste)
    """
    # DirectForecaster: dict {h: model}
    if hasattr(model, "models_") and getattr(model, "models_", None):
        models_dict = model.models_
        if hasattr(model, "horizons"):
            return [models_dict[h] for h in model.horizons if h in models_dict]
        return list(models_dict.values())

    # MIMO: MultiOutputRegressor(...).estimators_
    if hasattr(model, "model_") and hasattr(model.model_, "estimators_") and model.model_.estimators_:
        return list(model.model_.estimators_)

    return []


# =============================================================================
# Feature Importance (über alle Basis-Estimatoren mitteln)
# =============================================================================
def feature_importance_df(model, feature_cols: List[str]) -> pd.DataFrame:
    """
    Aggregiert Feature Importances über alle Basis-Estimatoren (entspricht allen H):
      - importance_mean / importance_std je Feature
    Fällt defensiv zurück, falls keine Importances verfügbar sind.
    """
    ests = get_estimators(model)
    if not ests:
        return _empty_fi_df(feature_cols)

    mats: List[np.ndarray] = []
    for est in ests:
        fi = getattr(est, "feature_importances_", None)
        if fi is None:
            continue
        mats.append(np.asarray(fi, dtype=float))

    if not mats:
        return _empty_fi_df(feature_cols)

    arr = np.vstack(mats)  # shape: (n_estimators, n_features)
    if arr.shape[1] != len(feature_cols):
        clog("FI", f"Featurezahl inkonsistent: importances={arr.shape[1]} vs features={len(feature_cols)} – positionsbasiert weiter.")
        # Weiter positionsbasiert; Feature-Reihenfolge = Trainingsreihenfolge

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": arr.mean(axis=0),
        "importance_std":  arr.std(axis=0),
    })
    return df.sort_values("importance_mean", ascending=False, ignore_index=True)


def _empty_fi_df(feature_cols: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({"feature": list(feature_cols),
                         "importance_mean": 0.0,
                         "importance_std": 0.0})


# =============================================================================
# SHAP (über alle Basis-Estimatoren mitteln)
# =============================================================================
def shap_values_aggregate(
    model,
    df_sample: pd.DataFrame,
    feature_cols: List[str],
    sample_rows: int = 1024,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Berechnet **aggregierte SHAP-Werte über alle Basis-Estimatoren** (über alle H).
    Rückgabe: (X_rep, shap_vals) mit identischer Zeilenzahl.
    """
    try:
        import shap  # optional
    except Exception:
        clog("SHAP", "shap nicht installiert – überspringe.")
        return df_sample.loc[:, feature_cols].iloc[:0], np.empty((0, len(feature_cols)), dtype=float)
    

    ests = get_estimators(model)
    if not ests:
        return df_sample.loc[:, feature_cols].iloc[:0], np.empty((0, len(feature_cols)), dtype=float)
    
    # Early-Exit: kein Sampling gewünscht → leere Rückgabe (spart Explainer-Overhead)
    if sample_rows <= 0:
        return df_sample.loc[:, feature_cols].iloc[:0], np.empty((0, len(feature_cols)), dtype=float)

    # 1) Subsample (deterministisch)
    X = df_sample.loc[:, feature_cols].copy()
    # -- ensure numeric for SHAP --
    for c in X.columns:
        if str(X[c].dtype) == 'category':
            X[c] = X[c].cat.codes.astype('int32')
        elif X[c].dtype == 'bool':
            X[c] = X[c].astype('int8')
        elif pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_datetime64_any_dtype(X[c]):
            codes, _ = pd.factorize(X[c], sort=True)
            X[c] = pd.Series(codes, index=X.index).replace({-1: np.nan}).astype("float32")

    if len(X) > sample_rows:
        X = X.sample(n=sample_rows, random_state=random_state)

    # 2) SHAP je Estimator (DataFrame an SHAP übergeben!)
    shap_blocks: List[np.ndarray] = []
    n_ok = 0
    for est in ests:
        try:
            # expl = shap.TreeExplainer(est)
            # vals = expl.shap_values(X)  # <-- KEIN .to_numpy(...)
            # schneller/ruhiger, v.a. für xgboost (gpu_hist) – mit Fallback für ältere SHAP-Versionen
            try:
                expl = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
                vals = expl.shap_values(X, check_additivity=False)  # <-- KEIN .to_numpy(...)
            except TypeError:
                expl = shap.TreeExplainer(est)
                vals = expl.shap_values(X)
            
            if isinstance(vals, list):
                # z.B. Multiclass → über Klassen mitteln
                vals = np.asarray(vals).mean(axis=0)
            vals = np.asarray(vals, dtype=float)
            if vals.ndim != 2 or vals.shape[1] != X.shape[1]:
                raise ValueError(f"SHAP-Shape unerwartet: {vals.shape} vs features={X.shape[1]}")
            shap_blocks.append(vals)
            n_ok += 1
        except Exception as e:
            clog("SHAP", f"Estimator ausgelassen ({type(est).__name__}): {e}")
            continue

    if n_ok == 0:
        return X.iloc[:0], np.empty((0, X.shape[1]), dtype=float) # iloc für stabiles Schema, verhindert key-errors wenn X leer ist

    # 3) vertikal stacken -> (n_estimators * n_samples, n_features)
    shap_vals = np.vstack(shap_blocks)

    # 4) X entsprechend wiederholen (gleiche Zeilenzahl wie shap_vals)
    X_rep = pd.concat([X] * n_ok, ignore_index=True)

    return X_rep, shap_vals

# =============================================================================
# Dataset-Spezifikation 
def shap_summary_df(
    model,
    df_sample: pd.DataFrame,
    feature_cols: List[str],
    sample_rows: int = 1024,
    random_state: int = 123,
) -> Optional[pd.DataFrame]:
    """
    Kompakte SHAP-Zusammenfassung „über alle H“:
    - nutzt shap_values_aggregate(...) (stacked über alle Estimatoren)
    - berechnet Mean(|SHAP|) je Feature über die **aggregierte** Matrix

    Rückgabe-DF (sortiert):
      ['feature', 'shap_mean_abs']
    """
    X_rep, shap_vals = shap_values_aggregate(
        model, df_sample, feature_cols, sample_rows=sample_rows, random_state=random_state
    )
    if shap_vals.size == 0:
        return None

    mean_abs = np.abs(shap_vals).mean(axis=0)  # über alle Estimatoren×Samples
    df = pd.DataFrame({"feature": feature_cols, "shap_mean_abs": mean_abs})
    return df.sort_values("shap_mean_abs", ascending=False, ignore_index=True)

# =============================================================================
# SHAP-Beeswarm (über alle H)
def shap_beeswarm_figure(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    *,
    max_display: int = 25,
    outdir: str | Path | None = None,
    filename: str = "shap_beeswarm_allH.png",
) -> Optional[Path]:
    """
    Stabiles Beeswarm (± um 0) **über alle Horizonte**.
    Erwartet die mit shap_values_aggregate(...) gestackten SHAP-Werte und das
    entsprechend replizierte X_df (gleiche Zeilenzahl); falls nicht, wird X_df
    defensiv auf die nötige Zeilenzahl erweitert.
    """
    try:
        import shap, matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        clog("SHAP", f"import failed: {e}")
        return None

    if shap_values.size == 0 or X_df.empty:
        return None

    outdir = Path(outdir) if outdir is not None else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename

    try:
        # --- Zeilenzahl ausgleichen (falls aggregierte SHAPs länger als X_df) ---
        if shap_values.shape[0] != len(X_df):
            reps = int(np.ceil(shap_values.shape[0] / max(1, len(X_df))))
            X_expanded = pd.concat([X_df] * reps, ignore_index=True).iloc[: shap_values.shape[0], :]
        else:
            X_expanded = X_df

        # >>> Farbwerte normalisieren (wichtig gegen "graue" Punkte):
        # Alle Spalten numerisch und ohne NaN, nur fürs Plotten!
        X_plot = X_expanded.copy()
        for col in X_plot.columns:
            s = X_plot[col]
            if pd.api.types.is_categorical_dtype(s.dtype):
                # kategorisch -> Codes; -1 (NaN) wird zu NaN und danach gefüllt
                s = s.cat.codes.replace({-1: np.nan}).astype("float64")
            elif pd.api.types.is_object_dtype(s.dtype):
                # Strings o.ä. -> factorize
                codes, _ = pd.factorize(s, sort=True)
                s = pd.Series(codes, index=s.index).replace({-1: np.nan}).astype("float64")
            elif pd.api.types.is_integer_dtype(s.dtype):
                # inkl. pandas' nullable Int64
                s = s.astype("float64")
            else:
                s = pd.to_numeric(s, errors="coerce")

            if s.isna().any():
                mu = float(np.nanmean(s))
                s = s.fillna(mu if np.isfinite(mu) else 0.0)

            X_plot[col] = s
        # <<< Ende Normalisierung

        # --- Plotten (bewährte "alte" API) ---
        plt.close("all")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_plot,
            max_display=max_display,
            plot_type="dot",
            show=False,
        )

        fig = plt.gcf()
        fig.suptitle("SHAP Beeswarm (über alle Horizonte)", fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.94)  # weniger Platz für den Titel

        fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return outpath

    except Exception as e:
        clog("SHAP", f"save failed: {e}")
        return None

# =============================================================================
# RMSSE pro Serie (ids in meinem Fall) (über alle ausgewerteten Zeilen und Horizonte)
# =============================================================================
def rmsse_per_series_df(
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame,
    *,
    series_index: pd.Index | pd.MultiIndex,
    scales: pd.Series | dict,         # <— dict auch zulassen
    series_cols: List[str],
    keys_df: Optional[pd.DataFrame] = None,
    id_col: str = "id",
    item_col: str = "item_id",
    store_col: str = "store_id",
) -> pd.DataFrame:
    # Numerator: mittlere quadratische Fehler pro Serie (über Zeilen *und* Horizonte)
    err_sq = (Y_pred - Y_true) ** 2
    mse_by_series = err_sq.groupby(series_index).mean().mean(axis=1)

    # Denominator (Skalen^2) – dict ODER Series robust behandeln
    scl = scales if isinstance(scales, pd.Series) else pd.Series(scales)
    denom = scl.reindex(mse_by_series.index).astype(float).pow(2).replace(0.0, np.nan)

    rmsse = np.sqrt(mse_by_series / denom)

    out = rmsse.reset_index()
    out.columns = list(series_cols) + ["rmsse"]  # z. B. ["id","rmsse"]

    # Optional: Meta-Spalten (item_id, store_id) anreichern
    if keys_df is not None:
        merge_key = id_col if id_col in out.columns else (series_cols[0] if series_cols else None)
        if merge_key and (merge_key in keys_df.columns):
            have_item  = item_col  in keys_df.columns
            have_store = store_col in keys_df.columns
            if have_item or have_store:
                cols = [merge_key] + ([item_col] if have_item else []) + ([store_col] if have_store else [])
                meta = keys_df.loc[:, cols].drop_duplicates(subset=[merge_key]).reset_index(drop=True)
                out = out.merge(meta, on=merge_key, how="left")
                # hübsche Reihenfolge
                order = [merge_key]
                if have_item:  order.append(item_col)
                if have_store: order.append(store_col)
                order.append("rmsse")
                out = out[order]

    return out.sort_values("rmsse", ascending=False, ignore_index=True)


# =============================================================================
# Compute-Backend erkennen (CPU/GPU + Detail)
# =============================================================================
def detect_compute_backend(model_name: str, model, params_in_effect: dict | None = None) -> tuple[str, str]:
    """
    Liefert ('gpu'|'cpu', detail_string) anhand Modellobjekt + Parametern.
    Robust für unsere Wrapper:
      - LGBM: DirectForecaster / MIMO (MultiOutputRegressor)
      - XGB : DirectForecasterXGB / MIMO
      - Darts (N-HiTS/TFT): nutzt torch.cuda.is_available()
      - GNS: CPU
    """
    m = (model_name or "").lower()

    def _first_estimator_from_wrapper(mdl):
        # Direct (dict mit Modellen) -> erstes Modell
        if hasattr(mdl, "models_") and isinstance(getattr(mdl, "models_", None), dict) and mdl.models_:
            try:
                return next(iter(mdl.models_.values()))
            except Exception:
                pass
        # MIMO: sklearn.MultiOutputRegressor
        if hasattr(mdl, "model_"):
            mo = getattr(mdl, "model_", None)
            # bevorzugt first estimator aus estimators_
            ests = getattr(mo, "estimators_", None)
            if isinstance(ests, (list, tuple)) and len(ests) > 0:
                return ests[0]
            # oder Basisschätzer-Vorlage
            est = getattr(mo, "estimator", None)
            if est is not None:
                return est
        # Fallback: gib selbst zurück
        return mdl

    def _params_of(est):
        try:
            if hasattr(est, "get_params"):
                return est.get_params(deep=False) or {}
        except Exception:
            pass
        # LightGBM: Booster-Params?
        try:
            booster = getattr(est, "booster_", None)
            if booster is not None and hasattr(booster, "params"):
                return dict(getattr(booster, "params", {}) or {})
        except Exception:
            pass
        return {}

    # --- Tree Ensembles ----------------------------------------------------
    if m in ("lightgbm", "lgbm"):
        est = _first_estimator_from_wrapper(model)
        p = {}
        p.update(_params_of(est))
        # zusätzlich: übergebene params_in_effect (können 'device_type' etc. tragen)
        if isinstance(params_in_effect, dict):
            q = {k: params_in_effect.get(k) for k in ("device_type","gpu_platform_id","gpu_device_id")}
            p.update({k: v for k, v in q.items() if v is not None})

        dev = str(p.get("device_type", "")).lower()
        if dev == "gpu" or ("gpu_platform_id" in p or "gpu_device_id" in p):
            return "gpu", "lightgbm(device_type=gpu)"
        return "cpu", f"lightgbm(tree_method=hist/CPU)"

    if m in ("xgboost", "xgb"):
        est = _first_estimator_from_wrapper(model)
        p = _params_of(est)
        # merge: in-effect params bevorzugen (device ab XGB>=2.0)
        if isinstance(params_in_effect, dict):
            p.update({k: v for k, v in params_in_effect.items() if v is not None})

        dev = str(p.get("device", "")).lower()
        tree_method = str(p.get("tree_method", "")).lower()
        predictor = str(p.get("predictor", "")).lower()
        if dev == "cuda" or tree_method == "gpu_hist" or predictor == "gpu_predictor":
            detail = "xgboost(device=cuda)" if dev == "cuda" else f"xgboost(tree_method={tree_method or 'gpu_hist'})"
            return "gpu", detail
        return "cpu", f"xgboost(tree_method={tree_method or 'hist'})"

    # --- Deep Learning (Darts / Torch) -------------------------------------
    if m in ("nhits", "n-hits", "tft", "temporal_fusion_transformer"):
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "gpu", "torch(cuda_available=True)"
            return "cpu", "torch(cuda_available=False)"
        except Exception:
            return "cpu", "torch(not_imported)"

    # --- Naive / GNS -------------------------------------------------------
    if m in ("gns", "naive", "seasonal_naive"):
        return "cpu", "global_naive_seasonal"

    # Fallback
    return "cpu", m or "unknown"

# =============================================================================
def shap_kernel_for_dl(model, df_ref: pd.DataFrame, feat_cols, *,
                       horizon_col="y+1",
                       background_n=200,
                       explain_n=200,
                       random_state=42):
    """
    Erklärt DL-Modell (TFT/N-HiTS) für einen einzelnen Horizon (default: 'y+1')
    via SHAP KernelExplainer auf kleinem Sample.
    Gibt zurück: shap_df (mean/std), X_explain (DataFrame), shap_values (ndarray).
    """
    rng = np.random.default_rng(random_state)

    # 1) explain-Subset ziehen
    if explain_n > len(df_ref):
        explain_n = len(df_ref)
    idx_explain = rng.choice(df_ref.index.to_numpy(), size=explain_n, replace=False)
    X_explain = df_ref.loc[idx_explain, feat_cols].astype(np.float32).copy()

    # 2) Background (Median-Profil oder kleine Stichprobe)
    if background_n > len(df_ref):
        background_n = len(df_ref)
    idx_bg = rng.choice(df_ref.index.to_numpy(), size=background_n, replace=False)
    X_bg = df_ref.loc[idx_bg, feat_cols].astype(np.float32).copy()

    # 3) Modell-Wrapper: mappt Matrix -> Vorhersagevektor für gewünschten Horizon
    def _f(X):
        X = pd.DataFrame(X, columns=feat_cols, index=idx_explain[:len(X)])  # gleiche Spalten
        # Klone der Referenz-Zeilen für sauberen Merge:
        tmp = df_ref.loc[idx_explain[:len(X)], :].copy()
        tmp.loc[:, feat_cols] = X.values
        # Vorhersage holen und gewünschten Horizon herausziehen:
        Yp = model.predict(tmp)
        return Yp[horizon_col].to_numpy(dtype=np.float64, copy=False)

    expl = shap.KernelExplainer(_f, X_bg, link="identity")
    # WICHTIG: kleine nsamples, sonst langsam
    shap_values = expl.shap_values(X_explain, nsamples=200)

    # 4) Aggregation
    abs_mean = np.mean(np.abs(shap_values), axis=0)
    abs_std  = np.std(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({
        "feature": feat_cols,
        "shap_mean": abs_mean,
        "shap_std": abs_std,
    }).sort_values("shap_mean", ascending=False).reset_index(drop=True)

    return shap_df, X_explain, shap_values

def dl_feature_importance_from_shap(shap_df: pd.DataFrame, top_k: int | None = None):
    out = shap_df.rename(columns={"shap_mean": "importance_mean", "shap_std": "importance_std"})[["feature","importance_mean","importance_std"]]
    if top_k:
        out = out.head(int(top_k))
    return out

# Ende model_utils.py
