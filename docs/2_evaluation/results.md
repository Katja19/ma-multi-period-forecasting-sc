## Zentrale Beobachtungen

- **Gesamtleistung**: Native Multi-Step DL (v. a. TFT, gefolgt von N-HiTS) erzielt die besten RMSSE_total-Werte.
- **Horizont-Verhalten:** DL-Modelle steigen über den Horizont moderat an; Tree-Ensembles (XGBoost/LightGBM) degradieren stärker ab mittleren Horizonten (≈ h 12–14).
- **Baselines**: GNS (s=7) liefert eine robuste Untergrenze und bestätigt die starke Wochenstruktur, wird jedoch von allen trainierten Modellen klar übertroffen.
- **Tuning**: HPO verbessert die Tree-Modelle leicht, ändert aber die Rangfolge nicht.
- **Robustheit**: Ergebnisse sind konsistent über beide Datensätze; Unterschiede zeigen sich vor allem im Horizontprofil $RMSSE_{h}$, weniger in der Grundtendenz.

--- 

## Wandb Report 


<iframe src="https://wandb.ai/katja-gegg/multi-period-forecasting/reports/Multi-Period-Forecasting-Report-Overall--VmlldzoxNDM5NTgxNg" style="border:none;height:1024px;width:100%">
