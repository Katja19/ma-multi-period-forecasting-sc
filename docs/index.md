<!-----
title: Master Thesis - MPF - Docs
template: home.html
---

DAS IST DER INDEX!-->

# Masterarbeit: Mehrperiodisches Forecasting für dynamische Entscheidungsfindung

## Abstract

Planungsentscheidungen erfordern Vorhersagen nicht nur für die nächste, sondern für mehrere kommende Perioden. Die vorliegende Masterarbeit untersucht, wie Machine Learing Modelle, die für Single-Step-Vorhersagen konstruiert wurden, mithilfe von Wrapper-Strategien mehrere Zeitpunkte (Multi-Step) in die Zukunft prognostizieren können und vergleicht diese mit Deep-Learning Modellen, die dazu native in der Lage sind. Anhand einer vergleichenden Taxonomie- und Evidenz-Matrix werden methodische Unterschiede sowie Vor- und Nachteile der bekanntesten Methoden analysiert und bewertet, auch im Vergleich zu den klassischen statistischen Modellen. Im zweiten Teil der Arbeit werden für zwei Datensätze aus dem Bereich des Retail-Marktes durch ein einheitliches Vorgehen native Modelle (TFT, N-HiTs) mit Wrapper-Modellen(LightGBM, XGBoost mit MIMO- und Direct-Wrappern) anhand der Performance-Metrik RMSSE und Rolling-Origin/Walk-Forward-Backtest verglichen. 

## Forschungsfragen

- **RQ1 (Modellfamilien über Horizonte):**  
    Wie unterscheiden sich Wrapped GBDTs (LightGBM, XGBoost; Direct & MIMO) und native Multi-Step-DL (TFT, N-HiTS) unter identischen, retail-nahen Bedingungen hinsichtlich RMSSE, getrennt nach kurzen (≤ 7) und *mittleren* (8–28) Horizonten?

- **RQ2 (Effizienz/Tuning-Trade-off):**  
    Welchen Mehrwert bringt Hyperparameter-Optimierung (HPO) für GBDT-Wrapper relativ zur Rechenzeit (Genauigkeitsgewinn pro Laufzeit), verglichen mit DL-Baselines mit schlankem Training?

- **RQ3 (Datensatz-Sensitivität innerhalb einer Domäne):**  
    Wie stabil ist die relative RMSSE-Performance der Modelle zwischen zwei Retail-Datensätzen gleicher Domäne (M5 Walmart vs. Bakery) mit unterschiedlichen Nachfrageeigenschaften?
