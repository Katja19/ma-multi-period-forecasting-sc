# datasets.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Mapping, Iterable, List, FrozenSet, Dict
import pandas as pd

__all__ = ["DatasetSpec", "get_spec", "detect_dataset_from_cols", "split_features_for_dl"]

# ==============================
# Dataset-Spezifikation
# ==============================
@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """
    Beschreibt ein Dataset und kapselt:
      - Normalisierung (Umbenennungen, Pflichtspalten, Datums-Casting)
      - Feature-Selektion
    Nutze direkt:
        spec = get_spec("bakery")
        df = spec.normalize(df)
        X_cols = spec.feature_columns(df, extra_drop=["foo", "bar"])
    """
    name: str
    date_col: str
    target_col: str
    series_cols: Tuple[str, ...]
    forbidden_exact: FrozenSet[str] = frozenset()
    forbidden_prefixes: Tuple[str, ...] = tuple()
    renames: Mapping[str, str] = field(default_factory=dict)  # z.B. {"sales": "demand"}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet optionale Umbenennungen an, prüft Pflichtspalten und castet das Datum.
        """
        df = df.rename(columns=self.renames) if self.renames else df.copy()
        needed = {self.date_col, self.target_col, *self.series_cols}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"[{self.name}] Missing columns: {sorted(missing)}")
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def feature_columns(self, df: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> List[str]:
        """
        Wählt Feature-Spalten X aus:
          - drop: Datum, Ziel, Serien-Keys
          - drop: y+H Targets (Spaltennamen beginnend mit 'y+')
          - drop: exakt verbotene Spalten (forbidden_exact)
          - drop: Präfix-Verbote (forbidden_prefixes)
          - optional: extra_drop
        Implementierung ist vektorisiert (ein Pass über die Spalten).
        """
        cols = pd.Index(df.columns)

        # Basisausschlüsse
        drop = set(self.series_cols) | {self.date_col, self.target_col} | set(self.forbidden_exact)
        if extra_drop:
            drop |= set(extra_drop)

        # String-sichere Serie der Spaltennamen für startswith-Checks
        s = pd.Series(list(cols), dtype="object")

        # y+ Targets
        yplus_mask = s.str.startswith("y+", na=False)

        # Präfix-Verbote (tuple unterstützt mehrere Präfixe)
        if self.forbidden_prefixes:
            prefix_mask = s.str.startswith(self.forbidden_prefixes, na=False)
        else:
            prefix_mask = pd.Series(False, index=s.index)

        # Keep/Drop-Maske
        mask_keep = ~cols.isin(drop) & ~yplus_mask.to_numpy() & ~prefix_mask.to_numpy()
        return cols[mask_keep].tolist()

# ==============================
# Dataset-Profile
# ==============================
_COMMON_FORBIDDEN: FrozenSet[str] = frozenset({
    "label", "scalingValue", "yearMonth", "yearWeek",
    "demand_original_reconstructed", "store_id", "item_id",
})

BAKERY = DatasetSpec(
    name="bakery",
    date_col="date",
    target_col="demand",
    series_cols=("id",),
    forbidden_exact=_COMMON_FORBIDDEN,
    forbidden_prefixes=("item_", "store_"),
)

M5 = DatasetSpec(
    name="m5",
    date_col="date",
    target_col="demand",
    series_cols=("id",),
    forbidden_exact=_COMMON_FORBIDDEN,
    # wenn du item_ behalten willst, nimm es aus der Liste
    forbidden_prefixes=("item_", "dept_", "cat_", "store_", "id_dummy_"),  # "state_" (3 states only)
)

# ==============================
# Registry + Getter
# ==============================
_REG: Dict[str, DatasetSpec] = {"bakery": BAKERY, "m5": M5}

def get_spec(name: str) -> DatasetSpec:
    try:
        return _REG[name]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(_REG)}")
    
# ==============================
# Feature-Utilities für DL-Wrapper (modellagnostisch)
# ==============================
from typing import Sequence, List
import re
import pandas as pd

def detect_dataset_from_cols(df: pd.DataFrame) -> str:
    """
    Heuristik wie bisher in models_dl: erkennt 'm5' anhand typischer Spaltenpräfixe,
    sonst 'bakery'. Funktional identisch zum bisherigen Code.
    """
    if any((c.startswith("is_snap_day") or c.startswith("state_")) for c in df.columns):
        return "m5"
    return "bakery"


def split_features_for_dl(
    all_cols: Sequence[str],
    dataset_name: str,
    mode: str = "generic",   # exakt wie bisher: 'generic' (N-HiTS) oder 'tft'
) -> tuple[List[str], List[str]]:
    """
    Teilt Feature-Spalten in future_known und past_only per Regex-Listen.
    Semantik identisch zum bisherigen _split_features_for_dl in models_dl.
    """
    cols = list(map(str, all_cols or []))
    dn = (dataset_name or "").lower()

    # --- GENERIC (robust, wie bisher) ---
    FUTURE_GENERIC = [
        r"^dayIndex$", r"^date$", r"^week$", r"^month$", r"^year$", r"^weekday$",
        r"^dayofyear$", r"(?:dayofyear|weekday|month)_(?:sin|cos)$",
        r"^weekday_[A-Z]{3}$", r"^month_[A-Z]{3}$", r"^year_\d{4}$",
        r"^is_.*event$", r"^is_(schoolholiday|holiday|holiday_next2days|snap_day)$",
        r"^promotion_currentweek$",
        r"^state_", r"^store_", r"^item_",
    ]
    PAST_GENERIC = [
        r"_lag_", r"_was_nan$", r"_diff_", r"_rolling_", r"_ratio_",
        r"__standard_deviation_", r"__variance_", r"__mean_", r"__median_",
        r"__sum_values_", r"__root_mean_square_", r"__maximum_", r"__minimum_",
        r"^demand(_|$)",
        r"^promotion_lastweek$",
        r"^rain(_|$)", r"^temperature(_|$)",
        r"^year_scaled$",
    ]
    if dn == "m5":
        FUTURE_GENERIC += [r"^state_", r"^weekday_[A-Z]{3}$", r"^month_[A-Z]{3}$", r"^year_\d{4}$"]

    # --- TIGHT für TFT (kleine, kuratierte Liste) ---
    FUTURE_TFT = [
        r"^dayofyear_(?:sin|cos)$",
        r"^weekday_(?:sin|cos)$",
        r"^month_(?:sin|cos)$",
        r"^is_.*event$",
        r"^state_",
        r"^promotion_currentweek$",
        r"^dayIndex$",
    ]
    PAST_TFT = [
        r"^demand_lag_(?:1|7|28)$",
        r"^demand_rolling_mean_(?:7|28)$",
    ]

    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        return any(re.search(p, name) for p in patterns)

    future_known, past_only = [], []
    if mode == "tft":
        for c in cols:
            if _matches_any(c, PAST_TFT):
                past_only.append(c)
            elif _matches_any(c, FUTURE_TFT):
                future_known.append(c)
    else:
        for c in cols:
            if _matches_any(c, PAST_GENERIC):
                past_only.append(c)
            elif _matches_any(c, FUTURE_GENERIC):
                future_known.append(c)
            else:
                past_only.append(c)

    fk = sorted(set(future_known))
    po = [c for c in sorted(set(past_only)) if c not in fk]
    return fk, po

# Ende datasets.py ==================================================================
