# core.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict, List

from config import (
    DATE_COL,               # z.B. "date"
    TARGET_COL,             # z.B. "demand"
    SERIES_COLS,            # z.B. ("id",)
    EVAL_HORIZONS_DEFAULT,  # z.B. (1, 7, 14, 28)
    N_TEST_ORIGINS_DEFAULT, # z.B. 56
    SEASONAL_PERIOD,        # z.B. 7
    CATEGORICAL_COLS,       # z.B. ("weekday", "month", "year", "week")
    VERBOSE,                # Logging-Schalter
    LOG_PREFIX,             # z.B. "CFG" (hier nicht zwingend, aber konsistent)
)

# __all__ nur noch das, was produktiv genutzt wird
__all__ = [
    "sort_df", "cast_categoricals", "count_series",
    "make_horizon_targets", "time_based_origin_split",
    "rmsse_scales", "rmsse_per_h_and_total",
    "rolling_train_val_windows",
    # "rmsse", "rmsse_by_horizon", "train_val_from_train",  # deprecated / unexported
]


# ---------------------------------------------------------------------
# Logging Helper
# ---------------------------------------------------------------------
def _log(msg: str) -> None:
    """Einfacher Print-Logger, schaltbar über config.VERBOSE."""
    if VERBOSE:
        print(msg)

# ---------------------------------------------------------------------
# Grund-Helper
# ---------------------------------------------------------------------
def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sortiert stabil nach Serien-Keys + Datum."""
    return df.sort_values(list(SERIES_COLS) + [DATE_COL], kind="mergesort").reset_index(drop=True) # kind="mergesort" macht den Sort stabil

def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Castet bekannte diskrete Zeitmerkmale (falls vorhanden) auf dtype 'category'.
    Das hilft LightGBM oft bei Speicher/Speed.
    """
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("category")
    return df

def count_series(df: pd.DataFrame) -> int:
    """Zählt eindeutige Zeitreihen gemäß SERIES_COLS (ein- oder mehrspaltig)."""
    if len(SERIES_COLS) == 1:
        return df[SERIES_COLS[0]].nunique(dropna=False)
    return df[list(SERIES_COLS)].drop_duplicates().shape[0]

# ---------------------------------------------------------------------
# Targets y+H bauen
# ---------------------------------------------------------------------
def make_horizon_targets(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT,
    *,
    assume_sorted: bool = False,
) -> pd.DataFrame:
    """
    Fügt für jeden Forecast-Horizont H eine Zielspalte 'y+H' hinzu.
    y+H ist jeweils der Wert des Targets H Tage in der Zukunft relativ zu t.

    Tipp: Wenn dein DF bereits nach SERIES_COLS und DATE_COL sortiert ist,
    setze assume_sorted=True, um das erneute Sortieren zu sparen.
    """
    df_out = df if assume_sorted else sort_df(df)
    df_out = df_out.copy()

    # Einmaliges GroupBy, dann pro Horizont nur noch shift – schnell & klar.
    gb_target = df_out.groupby(list(SERIES_COLS), sort=False, observed=False)[target_col]
    for h in horizons:
        df_out[f"y+{int(h)}"] = gb_target.shift(-int(h)).values
    return df_out

# ---------------------------------------------------------------------
# Zeitbasierter Split in Train/Test nach Forecast-Origins
# ---------------------------------------------------------------------
def time_based_origin_split(
    df_targets: pd.DataFrame,
    test_days: int = N_TEST_ORIGINS_DEFAULT,
    max_h: int = max(EVAL_HORIZONS_DEFAULT),
    *,
    assume_sorted: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Teilt nach Forecast-Origins: Train-Origins vor dem Testfenster,
    Test-Origins innerhalb des Testfensters.
    Filtert Zeilen, für die y+max_h nicht existiert.
    """
    df = df_targets if assume_sorted else sort_df(df_targets)
    # nur Zeilen behalten, wo der fernste Horizont verfügbar ist
    df = df.loc[~df[f"y+{int(max_h)}"].isna()]

    # Datumsgrenze für Testbereich (inklusive)
    last_date = df[DATE_COL].max()
    test_start = last_date - pd.Timedelta(days=int(test_days) - 1)
    
    # Clamp gegen früheste Origin
    origin_first = df[DATE_COL].min()
    if test_start < origin_first:
        test_start = origin_first

    train = df.loc[df[DATE_COL] < test_start].copy()
    test  = df.loc[df[DATE_COL] >= test_start].copy()
    
    
    _log("---- Data Split: core.py: time_based_origin_split() ----")

    # 1) Begriffe (einmal klarstellen)
    _log("Begriffe: Origins = Vorhersage-Starttage | Targets +h = Origins um h Tage nach rechts verschoben")

    # 2) Zusammenfassung der Fenster
    if not train.empty:
        _log(f"Origins train: {train[DATE_COL].min().date()} .. {train[DATE_COL].max().date()}")
    if not test.empty:
        tmin, tmax = test[DATE_COL].min(), test[DATE_COL].max()
        _log(f"Origins test : {tmin.date()} .. {tmax.date()}")
        _log(f"Anzahl Test-Origins (gewünscht/ist): {int(test_days)} / {test[DATE_COL].nunique()}")

        # Welche Horizonte sind tatsächlich im DataFrame vorhanden?
        used_horizons = sorted(
            int(str(c).split("+", 1)[1]) for c in df.columns
            if isinstance(c, str) and str(c).startswith("y+")
        )

        # 3) Ein „Aha“-Beispiel: erste Test-Origin → zugehörige Zieltage je Horizont
        first_origin = tmin
        example_map = ", ".join(
            f"+{h}:{(first_origin + pd.Timedelta(days=h)).date()}"
            for h in used_horizons
        )
        _log(f"Beispiel: erste Test-Origin {first_origin.date()} → Targets {example_map}")

        # 4) Ziel-Fenster je Horizont (Range)
        for h in used_horizons:
            if h in (1,2,7,14,27,28):
                tgt_start = (tmin + pd.Timedelta(days=h)).date()
                tgt_end   = (tmax + pd.Timedelta(days=h)).date()
                _log(f"Targets +{h}: {tgt_start} .. {tgt_end}")

        # 5) Orientierung: letzte Origin & letzter Zieltag für +max_h
        last_origin = tmax
        last_target_hmax = (last_origin + pd.Timedelta(days=int(max_h))).date()
        _log(f"Letzte Test-Origin: {last_origin.date()}  → letzter Target-Tag für +{int(max_h)}: {last_target_hmax}")

    # 6) Deine zwei sauberen Abschlusszeilen (unverändert)
    _log(f"Returns: Train: {train.shape[0]} rows, Test: {test.shape[0]} rows")
    _log(f"Train date: {train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}")
    _log(f"Test  date: {test[DATE_COL].min().date()} to {test[DATE_COL].max().date()}")


    return train, test

# ---------------------------------------------------------------------
# RMSSE: Skalen aus Train, Gesamt & je Horizont
# ---------------------------------------------------------------------
def rmsse_scales(
    train_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    seasonal_period: int = SEASONAL_PERIOD,
) -> pd.Series:
    """
    Berechnet pro Serie den Skalierungsfaktor S_i anhand der s-Lag-Differenzen
    im Train-Bereich (In-Sample). Rückgabe ist eine Series 'scale' indexiert
    nach SERIES_COLS.
    
    Vektorisierte Implementierung:
    diff_sq = (y_t - y_{t-s})^2
    scale_i = sqrt(mean(diff_sq)), 0 → NaN
    """
    m = int(seasonal_period)

    # Quadratische s-Lag-Differenzen je Gruppe (vektorisiert)
    g = train_df.groupby(list(SERIES_COLS), sort=False, observed=False)
    diff_sq = g[target_col].diff(periods=m)**2

    # Mittel pro Serie und sqrt; 0 → NaN, da RMSSE sonst unendlich/undefiniert wäre
    scale_sq_mean = (train_df.assign(__diff_sq__=diff_sq)
                     .groupby(list(SERIES_COLS), sort=False, observed=False)["__diff_sq__"]
                     .mean()
    )
    scales = np.sqrt(scale_sq_mean)
    scales = scales.where(scales > 0, np.nan)
    scales.name = "scale"
    
    return scales


#################################################
# --- unverändert lassen: rmsse_scales(...) aus A ---

def _rmsse_matrix(
    t: np.ndarray, p: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Vektorisierter Kern. Erwartet:
      t, p: shape (N, H) (bei H=1 auch (N,))
      s:    shape (N,)
    Liefert (per_h, total) mit per_h shape (H,).
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if t.shape != p.shape:
        raise ValueError(f"RMSSE: y_true/y_pred shape mismatch {t.shape} != {p.shape}")

    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape[0] != t.shape[0]:
        raise ValueError(f"RMSSE: scales length {s.shape[0]} != n_rows {t.shape[0]}")

    # Gültige Reihen (Skalen)
    valid_rows = np.isfinite(s) & (s > 0)
    # Gültige Ziel/Predict-Werte
    valid_tp = np.isfinite(t) & np.isfinite(p)

    err = p - t                              # (N, H)
    den = s.reshape(-1, 1)                   # (N, 1)

    # maskiere ungültige Reihen / Werte
    err = np.where(valid_rows[:, None], err, np.nan)
    den = np.where(valid_rows[:, None], den, np.nan)

    se = np.where(valid_tp, (err / den) ** 2, np.nan)  # (N, H)

    per_h = np.sqrt(np.nanmean(se, axis=0))            # (H,)
    total = float(np.sqrt(np.nanmean(se)))             # scalar
    return per_h, total


# Optional: öffentliche, bequeme Kombi-Funktion (B)
def rmsse_per_h_and_total(
    Y_true: pd.DataFrame,
    Y_pred: pd.DataFrame,
    series_index: pd.Index | pd.MultiIndex,
    scales: pd.Series
) -> tuple[dict[int, float], float]:
    """
    Vectorized: liefert (per_h_dict, total) in einem Durchlauf.
    """
    if isinstance(series_index, pd.MultiIndex) and series_index.nlevels == 1:
        series_index = series_index.get_level_values(0)

    cols = []
    hs = []
    for c in Y_true.columns:
        try:
            h = int(str(c).split("+", 1)[1])
            cols.append(c); hs.append(h)
        except Exception:
            continue
    order = [c for _, c in sorted(zip(hs, cols), key=lambda x: x[0])]
    hs_sorted = sorted(hs)

    t = Y_true.loc[:, order].to_numpy()
    p = Y_pred.loc[:, order].to_numpy()
    s = scales.reindex(series_index).to_numpy()

    per_h, total = _rmsse_matrix(t, p, s)
    return ({int(h): float(v) for h, v in zip(hs_sorted, per_h)}, float(total))

# ---------------------------------------------------------------------
def rolling_train_val_windows(
    train_df: pd.DataFrame,
    *,
    val_days: int,
    max_h: int,
    n_folds: int | None = None,
    step_days: int | None = None,
    min_train_days: int = 0,
    assume_sorted: bool = False,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Erzeugt Rolling-Origin-Folds (train_inner, val_df) nur aus dem TRAIN-Block.

    Parameter:
      - val_days:   Länge des Val-Fensters in Tagen (≈ Anzahl Origins).
      - max_h:      größter Horizont; Reihen ohne y+max_h werden entfernt.
      - n_folds:    optional. Wenn None, wird über step_days so viele Folds wie möglich gebaut.
      - step_days:  optional. Default = val_days (klassisches Sliding Window).
      - min_train_days: Mindestanzahl an Tagen, die train_inner vor val_start enthalten muss.
      - assume_sorted: falls train_df schon stabil sortiert ist.

    Rückgabe:
      Liste von (train_inner, val_df)-Tuplen, neueste Folds zuerst.
    """
    df = train_df if assume_sorted else sort_df(train_df)
    df = df.loc[~df[f"y+{int(max_h)}"].isna()].copy()
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    if df.empty:
        return folds

    # Basis-Zeiten
    first_date = df[DATE_COL].min()
    last_date  = df[DATE_COL].max()

    step = int(step_days) if step_days is not None else int(val_days)
    if step <= 0:
        raise ValueError("step_days must be >= 1 (oder None für Default=val_days)")

    # Wir starten beim letzten möglichen Val-Fenster (hinten) und gehen in Schritten zurück
    # val_end >= val_start, beide inklusive; val_days = Anzahl Kalendertage im Fenster
    # Beispiel: val_days=28  → val_start = last_date-27, val_end = last_date
    def window(i: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        val_end_i = last_date - pd.Timedelta(days=i * step)
        val_start_i = val_end_i - pd.Timedelta(days=int(val_days) - 1)
        return val_start_i, val_end_i

    i = 0
    built = 0
    while True:
        val_start_i, val_end_i = window(i)
        # Stop, wenn Fenster vollständig vor first_date läge
        if val_end_i < first_date:
            break
        # Clamp nach vorn
        if val_start_i < first_date:
            val_start_i = first_date

        # train_inner = alles strikt vor val_start_i
        tr = df.loc[df[DATE_COL] < val_start_i].copy()
        va = df.loc[(df[DATE_COL] >= val_start_i) & (df[DATE_COL] <= val_end_i)].copy()

        # Mindestanforderungen
        if va.empty or tr.empty:
            i += 1
            continue
        # Mindest-Train-Länge in Tagen (optional)
        if min_train_days > 0:
            train_span = (tr[DATE_COL].max() - tr[DATE_COL].min()).days + 1
            if train_span < int(min_train_days):
                i += 1
                continue

        folds.append((tr, va))
        built += 1
        if (n_folds is not None) and (built >= int(n_folds)):
            break
        i += 1

    _log("---- Data Split: core.py: rolling_train_val_windows() ----")
    _log(f"Gebaut: {len(folds)} Folds | val_days={int(val_days)} | step_days={int(step)} | min_train_days={int(min_train_days)}")
    for k, (tr, va) in enumerate(folds, start=1):
        _log(f"Fold {k:02d}: train {tr[DATE_COL].min().date()}..{tr[DATE_COL].max().date()}  | "
             f"val {va[DATE_COL].min().date()}..{va[DATE_COL].max().date()}  "
             f"(n_train_origins={tr[DATE_COL].nunique()}, n_val_origins={va[DATE_COL].nunique()})")
    return folds

# Ende core.py