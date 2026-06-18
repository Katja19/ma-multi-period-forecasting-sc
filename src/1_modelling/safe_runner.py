# safe_runner.py
from __future__ import annotations

import os    
from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, Any
from importlib import reload
from pathlib import Path
import pandas as pd

# from results_logger import append_run_summary # alte API
from results_logger import build_log_payload, append_log_payload # neue API

import config as CFG

ALLOWED_MODELS   = ("lightgbm", "xgboost", "nhits", "tft", "gns")
ALLOWED_WRAPPERS = ("direct", "mimo", "native", "baseline")
ALLOWED_DATASETS = ("bakery", "m5")
DEFAULT_EVAL_H = tuple(range(1, 29)) # (1..28)

# ===========================================================
# Konfiguration & Validierung
# ===========================================================
@dataclass(slots=True)
class ExperimentConfig:
    # --- User-facing ---
    run_suffix: str = "default"
    model: str = "lightgbm"                     # lightgbm | xgboost | nhits | tft | gns
    wrapper: str = "direct"                     # direct | mimo
    dataset: str = "bakery"                     # bakery | m5
    subset: bool = False
    use_hptuned_params: bool = False
    run_hpo: bool = False
    use_wandb: bool = True
    auto_save_hpo: bool = True
    save_csv_out_eval: bool = False

    # --- Horizons: User gibt nur eval_horizons; max_h wird intern = max(eval_horizons)
    eval_horizons: Sequence[int] = field(default_factory=lambda: DEFAULT_EVAL_H)

    # --- Sonstiges ---
    seasonal_period: int = 7
    shap_sample_rows: int = 1024

    # --- HW/Speed ---
    small_basics_when_not_tuned: bool = False   # optionale "kleine" Defaults für LGBM/XGB

    # --- GNS: explizites K (optional) ---
    gns_k: Optional[int] = None                 # empfohlen: 1..28

    # --- Data-Pfade (optional überschreiben) ---
    data_paths: Optional[Dict[str, Path]] = None

    # ----------------------------------------------------
    def normalize(self) -> None:
        self.model   = str(self.model).lower()
        self.wrapper = str(self.wrapper).lower()
        self.dataset = str(self.dataset).lower()
        self.eval_horizons = tuple(sorted({int(h) for h in self.eval_horizons}))
    
    # ----------------------------------------------------
    def validate(self) -> None:
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of {ALLOWED_MODELS}, got '{self.model}'")
        if self.wrapper not in ALLOWED_WRAPPERS:
            raise ValueError(f"wrapper must be one of {ALLOWED_WRAPPERS}, got '{self.wrapper}'")
        if self.dataset not in ALLOWED_DATASETS:
            raise ValueError(f"dataset must be one of {ALLOWED_DATASETS}, got '{self.dataset}'")
        if not self.eval_horizons:
            raise ValueError("eval_horizons must be non-empty")
        if any(int(h) < 1 for h in self.eval_horizons):
            raise ValueError("all eval_horizons must be >= 1")

        # Modell-spezifische Wrapper-Regeln
        m, w = self.model, self.wrapper
        if m in ("lightgbm", "xgboost"):
            if w not in ("direct", "mimo"):
                raise ValueError(f"for model='{m}' allowed wrappers are only 'direct' or 'mimo', got '{w}'")
        elif m == "gns":
            if w != "baseline":
                raise ValueError("for model='gns' the only allowed wrapper is 'baseline'")
        elif m in ("nhits", "tft"):
            if w != "native":
                raise ValueError(f"for model='{m}' the only allowed wrapper is 'native'")
            
        # NEU: Tuning bei TFT/NHiTS/GNS grundsätzlich verbieten
        _forbid_tuning_for_limited_models(self.model, self.use_hptuned_params, self.run_hpo)

        # HPO nur, wenn effektiver Wrapper 'mimo' ist (native/baseline zählen als mimo)
        eff_w = _effective_wrapper(self.model, self.wrapper)
        if self.run_hpo and eff_w != "mimo":
            raise ValueError("run_hpo=True requires an effective 'mimo' wrapper.")

        # Bei HPO W&B ruhigstellen
        if self.run_hpo and self.use_wandb:
            self.use_wandb = False
        
        # --- NEU: GNS-Konfiguration vereinfachen ---
        if m != "gns":
            if self.gns_k is not None:
                print("[INFO] gns_k angegeben, aber model!='gns' → wird ignoriert.")
        else:
            # Entweder explizites K ODER Tunings nutzen – exakt eines
            has_k  = self.gns_k is not None
            has_tp = bool(self.use_hptuned_params)
            if has_k and has_tp:
                raise ValueError("Für model='gns' gilt: Entweder 'gns_k' ODER 'use_hptuned_params=True', nicht beides.")
            if not has_k and not has_tp:
                raise ValueError("Für model='gns' musst du entweder 'gns_k' setzen oder 'use_hptuned_params=True' wählen.")
            if has_k:
                k = int(self.gns_k)
                if not (1 <= k <= 56):
                    raise ValueError("gns_k muss im Bereich 1..56 liegen (empfohlen: 1..28).")

    # kleine Defaults für LightGBM/XGBoost, wenn weder HPO noch Tunings
    def small_defaults_for_model(self) -> Optional[Dict[str, Any]]:
        if not self.small_basics_when_not_tuned:
            return None
        if self.model == "lightgbm":
            return dict(
                n_estimators=400, learning_rate=0.07, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=6, verbosity=-1,
            )
        if self.model == "xgboost":
            return dict(
                n_estimators=400, learning_rate=0.07, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                min_child_weight=1.0, n_jobs=6, random_state=42,
                verbosity=0, tree_method="hist",
            )
        return None

# ===========================================================
# Hilfsfunktionen (Daten)
# ===========================================================
#  Hilfsfunktion: effektiver Wrapper (für Tunings)
def _effective_wrapper(model: str, wrapper: str) -> str:
    m = (model or "").lower()
    w = (wrapper or "").lower()
    if m in ("gns", "nhits", "tft"):
        # diese Modelle laufen intern wie MIMO
        return "mimo"
    return w

def _default_data_paths(root: Path) -> Dict[str, Path]:
    return {
        "bakery": root / "data" / "basic_preprocessed" / "bakery_basic_prepro.csv",
        "m5":     root / "data" / "basic_preprocessed" / "m5_basic_prepro.csv",
    }

def _load_dataset(dataset: str, data_paths: Optional[Dict[str, Path]]) -> pd.DataFrame:
    root = Path.cwd().parents[1] if len(Path.cwd().parents) else Path.cwd()
    paths = data_paths or _default_data_paths(root)
    path = paths[dataset]
    df = (pd.read_csv(path)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .drop_duplicates(subset=["id", "date"])
            .sort_values(["id", "date"])
            .reset_index(drop=True))
    return df

def _subset_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "m5":
        return df[
            (df["item_id"].isin(["item_FOODS_3_080", "item_FOODS_3_226"])) &
            (df["store_id"].isin(["store_CA_1", "store_TX_1"]))
        ].copy()
    return df[
        (df["item_id"].isin([110, 109])) &
        (df["store_id"].isin([17, 19]))
    ].copy()
    

# Guard: Tuning-Verbot für TFT/NHiTS/GNS
def _forbid_tuning_for_limited_models(model: str, use_hptuned_params: bool, run_hpo: bool) -> None:
    m = (model or "").lower()

    # Deep Learning: zu rechenintensiv in dieser Umgebung
    if m in ("tft", "nhits"):
        if use_hptuned_params:
            raise ValueError(
                "use_hptuned_params=True ist für 'tft'/'nhits' nicht unterstützt "
                "(Grund: mangelnde Rechenressourcen)."
            )
        if run_hpo:
            raise ValueError(
                "run_hpo=True ist für 'tft'/'nhits' nicht unterstützt "
                "(Grund: mangelnde Rechenressourcen)."
            )

    # GNS: HPO/HPTuning ist überflüssig – einzig relevanter Parameter ist K
    if m == "gns":
        if use_hptuned_params or run_hpo:
            which = []
            if use_hptuned_params: which.append("use_hptuned_params")
            if run_hpo:            which.append("run_hpo")
            opts = " und ".join(which)
            raise ValueError(
                f"{opts} ist/sind für 'gns' nicht unterstützt: "
                "GNS hat außer 'K' keine tunebaren Hyperparameter – setze 'gns_k' explizit."
            )


# ============================================================
# Öffentliche API: Hauptfunktion
# ============================================================
def run_experiment(**kwargs) -> Dict[str, Any]:
    # 2) Safety: Env in der Funktion nochmal setzen (idempotent) + anzeigen
    CFG.force_cuda_device_env(gpu_id=CFG.PREFERRED_GPU_ID)
    print(f"[GPU ENV] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','unset')}")

    # 3) Reload der config, dann pipeline importieren & reloaden
    reload(CFG)  # liest aktuelle Defaults/Tunings unter gesetzter Env

    import pipeline as PIPE  # erst jetzt importieren!
    PIPE = reload(PIPE)

    from datasets import get_spec  # normale Imports – kein reload nötig

    # 4) Konfig bauen/validieren
    cfg = ExperimentConfig(**kwargs)
    cfg.normalize()
    cfg.validate()

    # 5) Globale CFG spiegeln wie gehabt
    CFG.DATASET_NAME = cfg.dataset
    CFG.DATASET_SPEC = get_spec(CFG.DATASET_NAME)
    CFG.USE_WANDB = cfg.use_wandb
    CFG.USE_HPTUNED_PARAMS = cfg.use_hptuned_params
    CFG.AUTO_SAVE_HPO = cfg.auto_save_hpo
    CFG.set_global_seed(CFG.RANDOM_STATE)

    # 6) Daten laden
    df = _load_dataset(cfg.dataset, cfg.data_paths)
    df_using = _subset_df(df, cfg.dataset) if cfg.subset else df

    # Horizons: max_h intern aus eval_horizons
    eval_h = tuple(int(h) for h in cfg.eval_horizons)
    max_h = int(max(eval_h))

    # Testfenster: NICHT user-settable → globaler Default aus config.py
    n_test_origins = int(CFG.N_TEST_ORIGINS_DEFAULT)

    # Modell-Params wählen
    model_params_for_run: Optional[Dict[str, Any]] = None
    if cfg.model == "gns":
        # nur noch zwei Fälle: explizites K → direkt setzen; sonst None, damit Tunings greifen
        if cfg.gns_k is not None:
            model_params_for_run = dict(K=int(cfg.gns_k), seasonal_period=cfg.seasonal_period)
        else:
            model_params_for_run = None
    else:
        if not cfg.run_hpo and not cfg.use_hptuned_params:
            model_params_for_run = cfg.small_defaults_for_model()

    
    
    eff_w = _effective_wrapper(cfg.model, cfg.wrapper)
    print("TUNED (effective):", (CFG.HPTUNED_PARAMS.get(cfg.model, {}).get(eff_w, {}).get(cfg.dataset)))

    # Wenn GNS + use_hptuned_params=True, sicherstellen, dass in der JSON etwas liegt
    if cfg.model == "gns" and cfg.use_hptuned_params:
        tuned = (CFG.HPTUNED_PARAMS
                 .get("gns", {})
                 .get(eff_w, {})
                 .get(cfg.dataset))
        if not tuned or ("K" not in tuned):
            raise ValueError(
                f"Keine getunten GNS-Parameter für ({cfg.dataset}, wrapper='{eff_w}') gefunden. "
                f"Bitte vorher HPO laufen lassen oder 'gns_k' explizit setzen."
            )
    
    # Pipeline starten
    result = PIPE.run_training(
        df_raw=df_using,
        df_name=cfg.dataset,
        model_name=cfg.model,
        wrapper=cfg.wrapper,
        max_h=max_h,                     # intern abgeleitet
        eval_horizons=eval_h,
        n_test_origins=n_test_origins,   # globaler Default
        run_name_suffix=cfg.run_suffix,
        log_wandb=cfg.use_wandb,
        seasonal_period=cfg.seasonal_period,
        model_params=model_params_for_run,
        feature_cols=None,
        shap_sample_rows=cfg.shap_sample_rows,
        use_hptuned_params=cfg.use_hptuned_params,
        run_hpo=cfg.run_hpo,
        subset=cfg.subset,
    )
    
    # Neue API: Log-Payload bauen + anhängen
    # Immer ein Log-Payload bereitstellen (für Notebook-Entscheidung)
    payload = build_log_payload(cfg, result)
    result["_log_payload"] = payload  # Notebook kann das später speichern
    
    # Alte API: nicht raus nehmen, da evtl. man es irgenwann vielleicht nutzen will
    # Optional: sofort schreiben, falls explizit gewünscht und „voller“ Lauf
    if cfg.save_csv_out_eval and (not cfg.subset) and (not cfg.run_hpo):
        print("[INFO] Appending run summary to CSV (save_csv_out_eval=True, no subset, no HPO).")
        written = append_log_payload(payload)
        if written:
            print("[INFO] Evaluation files:")
            for k, p in written.items():
                print(f"  - {k}: {p.resolve()}")
    else:
        reason = []
        if not cfg.save_csv_out_eval: reason.append("save_csv_out_eval=False")
        if cfg.subset: reason.append("subset=True")
        if cfg.run_hpo: reason.append("run_hpo=True")
        print(f"[INFO] Skipping CSV append ({', '.join(reason) or 'conditions not met'}).")
    
    return result


# ============================================================
# Hilfe-Text (für Notebook-Markdown)
# ============================================================
HELP_TEXT = f"""
    SAFE RUNNER – Kurzreferenz

    Parameter-Guide (safe_runner.ExperimentConfig)
    ----------------------------------------------
    • Modelle:   {ALLOWED_MODELS}
    • Wrapper:   {ALLOWED_WRAPPERS}
    • Datasets:  {ALLOWED_DATASETS}

    Wichtige Defaults
    -----------------
    • eval_horizons: Standard = 1..28 (RMSSE wird für diese Horizonte berechnet).
    • max_h wird intern als max(eval_horizons) gesetzt.
    • N Test-Origins: globaler Default aus config.py (CFG.N_TEST_ORIGINS_DEFAULT).
    • seasonal_period: 7
    • shap_sample_rows: 1024
    • use_wandb: True (wird bei HPO automatisch auf False gesetzt)
    • save_csv_out_eval: False

    Daten & Subsets
    ---------------
    • Pfade: Standard über _default_data_paths(<PROJECT_ROOT>):
    - bakery -> data/basic_preprocessed/bakery_basic_prepro.csv
    - m5     -> data/basic_preprocessed/m5_basic_prepro.csv
    • subset=True:
    - m5: item_id ∈ {{FOODS_3_080, FOODS_3_226}} UND store_id ∈ {{CA_1, TX_1}}
    - bakery: item_id ∈ {{110, 109}} UND store_id ∈ {{17, 19}}

    Wrapper/Modell-Kombinationen (Validierung)
    ------------------------------------------
    • lightgbm / xgboost:
    - Erlaubte Wrapper: "direct" oder "mimo"
    - HPO ist nur mit effektivem "mimo" möglich (siehe unten).
    • nhits / tft:
    - Erlaubter Wrapper: "native" (wird intern als "mimo" betrachtet)
    - Tuning/HPO grundsätzlich NICHT erlaubt (Rechenressourcen).
    • gns:
    - Erlaubter Wrapper: "baseline" (wird intern als "mimo" betrachtet)
    - HPO/Hyperparameter-Tunings NICHT erlaubt; einzig relevanter Parameter: K.
    - Du MUSST gns_k setzen (1..56, empfohlen 1..28).

    Effektiver Wrapper & HPO-Regel
    ------------------------------
    • _effective_wrapper(model, wrapper):
    - Für gns/nhits/tft -> "mimo"
    - Sonst -> Wrapper wie gewählt
    • run_hpo=True erfordert effektiven Wrapper "mimo".
    - Praktisch heißt das: HPO ist ausschließlich für lightgbm/xgboost mit wrapper="mimo" erlaubt.
    - Bei nhits/tft/gns ist HPO trotz effektivem "mimo" aus Ressourcengründen verboten.

    Tuning-Verbote (Sicherheits-Guards)
    -----------------------------------
    • nhits/tft:
    - use_hptuned_params=True -> Fehler
    - run_hpo=True -> Fehler
    • gns:
    - use_hptuned_params=True ODER run_hpo=True -> Fehler
    - setze stattdessen gns_k explizit (1..56).

    Kleine Defaults (Speed-Option)
    ------------------------------
    • small_basics_when_not_tuned=True aktiviert schnellere Defaults, ABER nur wenn
    weder run_hpo noch use_hptuned_params aktiv sind:
    - lightgbm: n_estimators=400, lr=0.07, num_leaves=63, subsample=0.8, colsample_bytree=0.8, ...
    - xgboost:  n_estimators=400, lr=0.07, max_depth=8, subsample=0.8, colsample_bytree=0.8, ...

    Logging & CSV-Ausgabe
    ---------------------
    • build_log_payload(cfg, result) wird immer erzeugt und unter result["_log_payload"] abgelegt.
    • append_log_payload(...) wird nur ausgeführt, wenn:
    - save_csv_out_eval=True
    - subset=False
    - run_hpo=False
    -> Ausgabe in <PROJECT_ROOT>/outputs/evaluation/*.csv

    W&B
    ---
    • use_wandb=True standardmäßig.
    • Bei run_hpo=True wird use_wandb automatisch auf False gesetzt (ruhigere Runs).

    Typische Fehlerquellen (und Lösungen)
    -------------------------------------
    1) run_hpo=True aber wrapper ist "direct":
    -> Fehler: HPO benötigt effektives "mimo". Lösung: wrapper="mimo".
    2) nhits/tft mit use_hptuned_params=True oder run_hpo=True:
    -> Nicht unterstützt. Lösung: beides False lassen.
    3) gns ohne gns_k:
    -> Für gns muss gns_k gesetzt werden (1..56). use_hptuned_params/run_hpo sind verboten.
    4) eval_horizons leer oder <1:
    -> eval_horizons muss nicht-leer sein und nur positive Integers enthalten.
    5) Falscher Wrapper zum Modell:
    -> z.B. xgboost mit "native" -> wird abgelehnt.

    Beispiele
    ---------
    • LightGBM, MIMO + HPO (erlaubt):
    run_experiment(model="lightgbm", wrapper="mimo", dataset="bakery",
                    eval_horizons=(1,7,14,28), run_hpo=True, use_wandb=False)

    • XGBoost, schnelle Defaults (kein HPO/Tuning):
    run_experiment(model="xgboost", wrapper="direct", dataset="m5",
                    small_basics_when_not_tuned=True)

    • GNS mit explizitem K:
    run_experiment(model="gns", wrapper="baseline", dataset="bakery",
                    gns_k=7, eval_horizons=(1,7,14,28))

    • N-HiTS/TFT (ohne Tuning/HPO):
    run_experiment(model="nhits", wrapper="native", dataset="m5",
                    eval_horizons=(1,7,14,28))

    Ablauf im run_experiment()
    --------------------------
    1) CUDA-Env setzen/anzeigen (CFG.force_cuda_device_env).
    2) config neu laden; pipeline importieren & reloaden.
    3) Dataset-Spezifikation spiegeln (CFG.DATASET_NAME / CFG.DATASET_SPEC).
    4) Konfiguration normalisieren & validieren (inkl. Guards).
    5) Daten laden (+ optional subset).
    6) max_h = max(eval_horizons); n_test_origins = CFG.N_TEST_ORIGINS_DEFAULT.
    7) Modell-Parameter:
    - gns: K explizit über gns_k; sonst Fehler.
    - lgbm/xgb: ggf. small_defaults (wenn kein HPO/Tuning).
    8) PIPE.run_training(...) starten.
    9) Log-Payload anhängen; optional CSV schreiben (Bedingungen siehe oben).

    Hinweis
    -------
    • Für große/aufwendige Modelle (nhits/tft) sind Tunings/HPO in dieser Umgebung
    absichtlich deaktiviert.
    """

def explain_params() -> str:
    return HELP_TEXT.strip()

# End of file =================================================