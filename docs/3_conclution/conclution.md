# Gesamtfazit
Die Arbeit zeigt: Es gibt keinen universellen Sieger im Multi-Step-Forecasting. Die relative Güte hängt systematisch von Horizont, Datenlage, verfügbaren known-future-Kovariaten sowie Metrik/Protokoll ab. Unter einheitlichen, retail-nahen Bedingungen ($H=28$, Walk-Forward mit 56 Origins) ergeben sich jedoch konsistente Muster.

## **Leistungsbild:** 
Native Multi-Step-Deep-Learning-Modelle (vor allem TFT, gefolgt von N-HiTS) erzielen die besten Gesamtfehler und flachere $RMSSE_{h}$-Profile. Tree-Ensembles (XGBoost/LightGBM) degradieren über längere Horizonte stärker; einzig auf M5 liegt XGBoost (getuned) knapp vor N-HiTS, ohne die Dominanz der DL-Modelle insgesamt zu kippen.

## **Effizienz & Tuning:** 
Hyperparameter-Optimierung bringt für Trees kleine, reproduzierbare Zugewinne (ca. 0,01–0,02 $RMSSE_{total}$), erhöht aber die Trainingszeit; die Rangfolge bleibt unverändert. In der gewählten Pipeline waren die DL-Läufe teils schneller (GPU, schlanke Presets).

## **Limitationen:** 
Ressourcenschonende Presets, zwei tägliche Retail-Datensätze und Punktprognosen (Quantile nicht genutzt) begrenzen die Generalisierbarkeit; die Befunde sind als konservative Untergrenze der Modellfamilien zu lesen.

## **Praxisleitfaden:**  
- **kurze Horizont**: $h ≤ 7$ & stark tabellarische Treiber → Trees (Direct/MIMO) als robuste, effiziente Wahl   
- **mittlerer bis langer Horizont**: $h ≥ 14$ und/oder verlässliche known-future-Kovariaten → TFT; ohne viele Exogene und für stabile Langhorizonte → N-HiTS    
- Fazit: Walk-Forward mit vielen Origins und $RMSSE_{h}$ ergänzend zu $RMSSE_{total}$ einsetzen

## **Fazit:** 
Für operative Mehrperioden-Entscheidungen liefern global trainierte, nativ multi-horizon-fähige DL-Modelle die höchste Prognosequalität, während GBDT-Wrapper wertvolle, effiziente Referenzen bleiben — insbesondere auf kurzen Horizonten.

