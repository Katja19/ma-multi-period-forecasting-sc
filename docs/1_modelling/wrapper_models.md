# XGBoost & LightGBM (Direct / MIMO)

## Kurzüberblick
**XGBoost** und **LightGBM** sind Gradient-Boosted-Tree-Modelle. Über **Wrapper** werden sie von einperiodischen Regressoren zu Multi-Step-Vorhersagern:

### Unterschied
- **Baumwachstum**: XGBoost wächst i. d. R. level-wise (ausbalanciert), LightGBM leaf-wise mit Tiefenlimit (aggressiver, oft genauer bei gleichem Baumumfang, aber potenziell überfitting-anfälliger ohne Regularisierung).  
- **Effizienz**: LightGBM nutzt GOSS (Gradient-based One-Side Sampling) und EFB (Exclusive Feature Bundling) → meist schneller und speichereffizienter bei vielen Features.  
- **Kategoriale Features**: Beide können native kategoriale Splits; in der Praxis ist LightGBM hier oft plug-and-play, während bei XGBoost in vielen Setups weiterhin One-Hot verbreitet ist.  
- **Tuning/Regularisierung**: XGBoost bietet sehr feingranulare Regularisierung (u. a. `gamma`, `lambda`, `alpha`), LightGBM skaliert stark über `num_leaves`/`max_depth` und Blatt-/Feature-Sampling.  
- **Praxis-Heuristik**: Starte bei breiten, hochdimensionalen Feature-Matrizen häufig mit LightGBM (Tempo/Memory). Für feines Overfitting-Controlling oder wenn Level-wise robuster ist, lohnt XGBoost.


## Wrapper
- **Direct:** ein separates Modell je Horizont ($h\in\{1,\dots,H\}$)  
- **MIMO:** ein gemeinsames Modell erzeugt alle ($H$) Ziele in einem Schritt (Multi-Output)

## Input-Design (tabellarisch)
- Ziel-Lags/Differenzen (z. B. 1/7/14/28), rollierende Kennzahlen (7/14/28), bekannte Zukunftskovariaten (Kalender/Events), kategoriale One-Hots (Store/Item/State).  
- Strikte Kausalität (keine Look-ahead-Lecks); NaN-Flags für gelaggte Features.

## Trainingsdetails (arbeitsspezifisch)
- **Backtest:** Rolling-Origin / Walk-Forward mit 56 Origins, ($H=28$).  
- **Validation/HPO:** zeitbasierte Folds (z. B. rollend); Ergebnisse von MIMO-HPO können als Startpunkt für Direct dienen.  
- **Metrik:** Reporting via $RMSSE_{h}$ und $RMSSE_{total}$.

## Typische Hyperparameter
- **Gemeinsam:** `max_depth`, `num_leaves` (LGBM), `n_estimators`, `learning_rate`, `min_child_weight`/`min_data_in_leaf`, `subsample`, `colsample_bytree`  
- **Direct-spezifisch:** ggf. leichter Regularisierungs-Bias pro ($h$) (Überanpassung vermeiden)  
- **MIMO-spezifisch:** Ziel-Vektorisierung/Output-Form prüfen; korrelierte Horizonte profitieren oft von MIMO

## Stärken & Grenzen
**Stärken:** sehr stark bei tabellarischen Treibern; schnelle Trainings-/Inference-Zeiten; Feature-Importance/SHAP verfügbar  
**Schwächen:** Performance nimmt über lange Horizonte oft schneller ab (v. a. Direct); Feature-Engineering ist entscheidend

