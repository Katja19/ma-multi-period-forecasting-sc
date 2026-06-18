# models_naive.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from config import DATE_COL, TARGET_COL, SERIES_COLS, EVAL_HORIZONS_DEFAULT

@dataclass
class GlobalNaiveSeasonal:
    horizons: Sequence[int] = EVAL_HORIZONS_DEFAULT
    model_params: Optional[Dict] = None
    seasonal_period: int = 7
    K: int = 1  # Anzahl vergangener Saisons zum Mittel (z. B. 1 oder 7)

    def __post_init__(self):
        self.horizons = tuple(int(h) for h in self.horizons)
        self.H = int(max(self.horizons))
        if self.model_params:
            self.seasonal_period = int(self.model_params.get("seasonal_period", self.seasonal_period))
            self.K = int(self.model_params.get("K", self.K))
        self._hist: Dict[str, pd.Series] = {}

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]) -> "GlobalNaiveSeasonal":
        # Merke Historie pro Serie als pd.Series (index=date → target)
        for gid, g in df_train.sort_values(list(SERIES_COLS)+[DATE_COL]).groupby(list(SERIES_COLS), sort=False):
            uid = gid if isinstance(gid, str) else "_".join(map(str, np.atleast_1d(gid)))
            s = g.set_index(DATE_COL)[TARGET_COL].astype(float)
            self._hist[uid] = s
        return self
    
    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        m = int(self.seasonal_period)
        Hs = tuple(int(h) for h in self.horizons)

        rows = []
        df_in = df_in.sort_values(list(SERIES_COLS) + [DATE_COL])

        for _, r in df_in.loc[:, list(SERIES_COLS) + [DATE_COL]].iterrows():
            uid = str(r[SERIES_COLS[0]]) if len(SERIES_COLS) == 1 else "_".join(map(str, [r[c] for c in SERIES_COLS]))
            origin = pd.to_datetime(r[DATE_COL])

            hist = self._hist.get(uid, pd.Series(dtype=float))
            # robust: sicherstellen, dass der Index Datetime ist (ohne Uhrzeit)
            try:
                if not np.issubdtype(hist.index.dtype, np.datetime64):
                    hist.index = pd.to_datetime(hist.index)
            except Exception:
                hist.index = pd.to_datetime(hist.index)
            # falls nötig: hist.index = hist.index.normalize(); origin = origin.normalize()

            row = {c: r[c] for c in list(SERIES_COLS) + [DATE_COL]}
            for h in Hs:
                rpos = ((h - 1) % m) + 1  # Position 1..m innerhalb der Saison
                vals = []
                # letztes bekanntes Datum ist origin-1; greife immer in die Vergangenheit:
                # t = origin + (rpos - 1) - k*m  ==  origin - (k*m - rpos + 1)
                for k in range(1, self.K + 1):
                    t = origin + pd.Timedelta(days=rpos - 1 - k * m)
                    if t in hist.index:
                        vals.append(hist.loc[t])
                row[f"y+{h}"] = float(np.mean(vals)) if vals else np.nan
            rows.append(row)

        out = pd.DataFrame(rows).set_index(df_in.index)
        return out[[f"y+{h}" for h in Hs]]

# Ende models_naive.py
