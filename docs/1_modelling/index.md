## Überblick 

Die folgende Tabelle fasst die in dieser Arbeit genutzten **Modellfamilien** und **Wrapper** zusammen.  
*Global/Lokal* bezieht sich auf Cross-Series-Pooling; *Uni/Multi* auf die verwendeten Eingaben (nur Ziel vs. mit Exogenen).

| Familie            | Variante                             | Wrapper / Output                         | Scope (Global/Lokal)                     | Input (Uni/Multi)                              | Kurznotiz |
|--------------------|--------------------------------------|------------------------------------------|------------------------------------------|-----------------------------------------------|-----------|
| **Baseline**       | **GNS** | Keiner – Kopie mit Lag *s=7* (Multi-H)   | Global                                    | Uni (optional Multi mit Kalender)              | Wochenmuster-Baseline |
| **Tree-Boosting**  | **LightGBM — Direct**                | Direct (ein Modell je Horizont *h*)      | Global (üblich), lokal möglich            | Uni **oder** Multi                             | flexibel, hoher Trainingsaufwand |
| **Tree-Boosting**  | **LightGBM — MIMO**                  | MIMO (ein Modell sagt alle *H*)          | Global (üblich), lokal möglich            | Uni **oder** Multi                             | konsistente Pfade, effizient |
| **Tree-Boosting**  | **XGBoost — Direct**                 | Direct (ein Modell je Horizont *h*)      | Global (üblich), lokal möglich            | Uni **oder** Multi                             | stark bei tabellarischen Treibern |
| **Tree-Boosting**  | **XGBoost — MIMO**                   | MIMO (ein Modell sagt alle *H*)          | Global (üblich), lokal möglich            | Uni **oder** Multi                             | gute Effizienz, stabile Outputs |
| **Deep (Seq.)**    | **N-HiTS**                           | Native Multi-Output über *H*             | Global (Standard)                         | Uni (nur Ziel) **oder** Multi (mit Kovariaten) | mehrstufige Dekomposition |
| **Deep (Seq.)**    | **TFT**| Native Multi-Output über *H*             | Global (Standard)                         | Multi (typisch; ohne Exogene „uni-like“ möglich)| Attention, known-future-Features |

> **Hinweise:**  
> • *Direct* = separates Modell für jeden Horizont \(h \in \{1,\dots,H\}\).  
> • *MIMO* = ein Modell erzeugt den gesamten Pfad \(1\dots H\) in einem Rutsch.  
> • *Uni* = nur Zielvergangenheit/zeitl. Ableitungen; *Multi* = inkl. exogener Kovariaten (Kalender/Events etc.).

---

## Gemeinsamer Trainings- & Evaluationsrahmen

- **Horizont:** \(H = 28\) Tage  
- **Backtest:** Rolling-Origin / Walk-Forward mit 56 Test-Starts  
- **Metriken:** $RMSSE_{h}$ (je Horizont) & $RMSSE_{total}$ (gesamt)  
- **Validierung:** DL mit zeitbasiertem Train/Val-Split (Early Stopping), Trees mit zeitbasierten Folds (HPO)

---

## Modelle

### 1) Naive Wochenbaseline (GNS)
Die Global Naive Seasonal (GNS)-Baseline setzt auf Wochensaisonalität mit Lag s=7 und optionaler Glättung:  
- **Vorhersage:** \(\hat{y}_{t+h} = y_{t+h-7}\) (ggf. gemittelt über mehrere 7er-Lags).  
- **Eigenschaften:** extrem robust, parameterarm, stark bei klaren Wochenmustern; dient als untere Referenzlinie in allen Vergleichen.

### 2) Wrapped Single-Step
Direct und MIMO umhüllen einperiodische Regressoren (z. B. XGB/LGBM), um Multi-Horizon-Pfade zu liefern:  
- **Direct:** ein separates Modell pro Horizont \(h\) (hohe Flexibilität, mehr Trainingsaufwand).  
- **MIMO:** ein gemeinsames Modell erzeugt alle \(H\) Schritte auf einmal (konsistente Pfade, effizienter).


### 3) Native Multi-Horizon DL (Kurzüberblick)
Sequenzmodelle lernen gemeinsame zeitliche Muster und known-future-Kovariaten end-to-end:  
- **TFT:** Attention + Gating, statische/zeitvariante/known-future Features, interpretierbar.  
- **N-HiTS:** mehrstufige Hierarchical Time-Series-Dekomposition mit direkter Vorhersage des gesamten Pfads.

---

## Feature-Nutzung (gemeinsam)
- **Known-future**: Kalender/Events (und je Datensatz spezifische Kovariaten).  
- **Vergangenheit**: Ziel-Lags/Differenzen, rollierende Kennzahlen (modellpfad-abhängig).  
- **Kategorisch**: Store/Item (One-Hot/Embedding).  

---

## Auswahl & Vergleich
- Modelle werden identisch gebacktestet und entlang $RMSSE_{total}$ und $RMSSE_{h}$-Profilen verglichen.  
- Die naive GNS dient als Qualitätsbar; „Gewinner“ unterscheiden sich je nach Horizont und Daten­lage.

---
