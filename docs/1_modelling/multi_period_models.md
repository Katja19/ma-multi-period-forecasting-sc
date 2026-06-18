
# Native Multi-Step Modelle: TFT & N-HiTS

## Kurzüberblick
**Temporal Fusion Transformer (TFT)** und **N-HiTS** sind nativ mehrhorizontfähige Sequenzmodelle, die den kompletten Vorhersagepfad in einem Schritt liefern (Multi-Output):

- **TFT:** fusioniert Vergangenheit, Gegenwart und known-future-Kovariaten via Embeddings, Gating, Variable Selection & Attention.  
- **N-HiTS:** direkte Pfadvorhersage über mehrstufige Hierarchical Time-Series-Dekomposition mit mehreren Heads.

## Input-Design (sequenziell)
- **Lookback-Sequenz** der Zielgröße (optional vergangene Kovariaten).  
- **Known-future**-Features für die nächsten ($H$) Tage (z. B. Kalender/Events/Promotions).  
- Kategorien (Store/Item/State) typischerweise als Embeddings; numerische Features skaliert.  
- Strikte Kausalität (kein Look-ahead); ggf. NaN-Flags für gelaggte Größen.

## Trainingsdetails (arbeitsspezifisch)
- **Backtest:** Rolling-Origin / Walk-Forward mit 56 Origins, ($H=28$).  
- **Validation:** zeitbasierter Train/Val-Split mit Early Stopping (Best Checkpoint).  
- **Metrik:** Reporting via $RMSSE_{h}$ und $RMSSE_{total}$.

## Typische Hyperparameter
**TFT**
- Lookback-/Decoder-Länge, Hidden/FFN-Size  
- # Blöcke / # Attention-Heads  
- Dropout, Batch Size, Learning Rate, Weight Decay  
- (Optional) Quantile für probabilistische Ausgaben

**N-HiTS**
- # Stacks / Blocks / Heads  
- Blockgrößen (Kernel/Pooling), Lookback-Länge  
- Hidden Units je Block  
- Dropout, Batch Size, Learning Rate

## Stärken & Grenzen
**TFT**  
- **Stärken:** sehr gute Nutzung von known-future-Kovariaten; teils interpretierbar (Variable Selection/Attention)  
- **Schwächen:** rechen-/speicherintensiver; sensibel auf Val-Setup/Regularisierung

**N-HiTS**
- **Stärken:** stabil über lange Horizonte, oft effizient; starke univariate Basis  
- **Schwächen:** weniger erklärbar; Vorteil sinkt bei sehr kurzen Horizonten/ohne Sequenzstruktur

