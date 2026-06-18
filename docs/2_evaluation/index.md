# Evaluation 

## Setup & Metriken

- **Backtest**: Rolling-Origin / Walk-Forward mit 56 Test-Starts, Horizont ($H=28$).
- **Primärmetrik**: RMSSE  
  - **$RMSSE_{total}$**: ein Gesamtwert über alle Serien, Origins und Horizonte.  
  - **$RMSSE_{h}$**: horizontspezifisch (Fehler pro Vorhersagetag ($h=1…28$)).

---

## RMSSE – Root Mean Squared Scaled Error

Im Kontext von Retail-Zeitreihen mit vielen Produkten und Hierarchieebenen hat sich der RMSSE als Standard etabliert.  
Er ist skalenunabhängig, nutzt eine in-sample Skala und ermöglicht so den Vergleich verschiedener Serien.  
RMSSE ist das quadrierte Gegenstück zum MASE.

**Definition (einzelne Serie):**

$$
\text{RMSSE} \;=\; \sqrt{
  \frac{
    \tfrac{1}{h}\,\sum_{t=n+1}^{n+h} (y_t - \hat{y}_t)^2
  }{
    \tfrac{1}{n-s}\,\sum_{t=s+1}^{n} (y_t - y_{t-s})^2
  }
}
$$

mit einem (saisonalem) Lag $s$. Übliche Wahl ist $s=1$ (klassische M5-Formel) oder bei täglichen Daten mit Wochenmuster $s=7$. **Wir verwendeten $s=7$.**

- **Zähler:** MSE der Vorhersagen im Auswertungsfenster $t=n+1,\dots,n+h$.  
- **Nenner:** In-sample Skala über das Trainingsfenster $t=1,\dots,n$: mittlere quadratische $s$-Lag-Differenz; äquivalent zum in-sample Fehler einer (seasonal-)naiven Prognose mit Lag $s$.  
- Wertebereich: $[0,\infty)$.  
- Interpretation: $\text{RMSSE}<1$ besser als (seasonal-)naiv, $\text{RMSSE}=1$ gleich gut, $\text{RMSSE}>1$ schlechter.

---

## **Aggregation in dieser Arbeit (mehrere Serien & Test-Origins)**

Das obige Prinzip bleibt erhalten; für die Berichte berechnen wir jedoch **horizontweise** Kennzahlen und eine **Gesamtzahl**:

- **Pro Horizont** $h$: Mittelung der **skalierten Quadrate** über **alle Serien** $i$ und **alle Test-Origins** $t$, **danach** Wurzel:

$$
\mathrm{RMSSE}_{h}
=
\sqrt{
\frac{1}{N_{\mathrm{series}}\,N_{\mathrm{origins}}}
\sum_{i=1}^{N_{\mathrm{series}}}
\sum_{t\in\mathcal{T}}
\frac{\bigl(y_{i,t+h}-\hat y_{i,t+h}\bigr)^2}{S_i}
},
\qquad
S_i=\frac{1}{n_i-s}\sum_{u=s+1}^{n_i}\bigl(y_{i,u}-y_{i,u-s}\bigr)^2.
$$


- **Gesamtwert** $\mathrm{RMSSE}_{\text{total}}$: Mittelung **aller** skalierten Quadrate über **alle** $h=1\dots H$, Serien und Origins, **vor** der Wurzel (also **nicht** der Mittelwert der $\mathrm{RMSSE}_h$):

$$
\mathrm{RMSSE}_{\mathrm{total}}
=
\sqrt{
\frac{1}{N_{\mathrm{series}}\,N_{\mathrm{origins}}\,H}
\sum_{i=1}^{N_{\mathrm{series}}}
\sum_{t\in\mathcal{T}}
\sum_{h=1}^{H}
\frac{\bigl(y_{i,t+h}-\hat y_{i,t+h}\bigr)^2}{S_i}
}.
$$

**Anmerkung**

- Die Wahl von $s$ muss *konsistent* für alle verglichenen Methoden sein (z. B. $s=7$ für tägliche Daten mit Wochenmuster).  
- Bei zu kurzer oder nahezu konstanter Serie kann der Nenner sehr klein werden; praktisch wird dann ein stabilisierender Fallback (z. B. Median-Skala über Serien) verwendet.



