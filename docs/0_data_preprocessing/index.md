# **Data Analysis und Preprocessing**

## Überblick

Im Rahmen dieser Arbeit werden zwei Datensätze betrachtet. Hierbei handelt es sich um einen Datensatz der Supermarktkette **Walmart**, einem Datensatz einer **Bäckerreikette**.

| Dataset       | Source       | Anuahl Produkte | Anzahl Läden  | Time Span               | Anzahl Tage (Wochen)   |
|---------------|--------------|-----------------|---------------|-------------------------|------------------------|
| bakery        | Universität   |  3              |  32           | 30.01.2016 — 30.04.2019 | 1186 Tage (ca. 169)    |
| M5 (Walmart)	| Univ./Kaggle | 10              |  10           | 26.02.2011 — 19.06.2016 | 1941 Tage (ca. 277)    |

## Zielgröße & Granularität

- Prognoseziel: **tägliche Nachfrage** (`demand`) je **Store × Produkt**.  
- Frequenz: **täglich**, gleiches Grundschema für beide Datensätze.  
- Alle Schritte (Cleaning, Features, Splits) sind **zeitkausal** ausgelegt (kein Look-ahead).

## Grundvorgehen

### 1. Data Collection
- Eingangsdaten laden, Versionierung dokumentieren, Rohdaten unangetastet archivieren.  
- Einheitliche Namenskonventionen und Datentypen (Datum, numerisch, kategorial).

### 2. Data Overview & Description
- Schema prüfen (Primärschlüssel, Vollständigkeit, Eindeutigkeit).  
- Basisstatistiken und erste Verteilungschecks (ohne tiefgehende Datensatzdetails).  
- Feature-Metadaten führen (Name, Typ, Herkunft „original/derived“, kurze Beschreibung).

### 3. Data Cleaning
- **Fehlende Werte**: prüfen; falls vorhanden → kausale Imputation oder Flagging.  
- **Dubletten/Inkonsistenzen**: entfernen bzw. harmonisieren (Datentypen, Wertebereiche).  
- **Ausreißer**: identifizieren; harte Messfehler entfernen, sonst **flaggen** bzw. **winsorisieren** (keine rückwirkende Glättung zukünftiger Informationen).

### 4. Exploratory Data Analysis (EDA)
- Zeitreihen-Basics: Trend/Saisonalität, Autokorrelation, Wochentagsmuster.  
- Korrelationen zwischen Ziel und erklärenden Variablen (nur generische Einsichten).  
- Ergebnisse dienen zur Hypothesenbildung fürs Feature Engineering.

### 5. Feature Engineering & Transformation
- **Zeitmerkmale**: Wochentag, Woche, Monat, Jahr; optional zyklische Kodierung.  
- **Lag-/Rolling-Features**: Lags/Differenzen und rollierende Kennzahlen über sinnvolle Fenster; zugehörige **NaN-Flags** für Kausalität.  
- **Kategorische Variablen**: One-Hot/Label/Target Encoding (je nach Modellfamilie).  
- **Skalierung**: robuste/normale Skalierung numerischer Variablen (Modellabhängigkeit beachten).  
- **Bekannte Zukunftskovariaten** (Kalender/Events etc.) werden als **known-future Features** modelliert.  
- Modellpfad-spezifisch:
  - **Tree-Ensembles** (Wrapped Single-Step, z. B. Direct/MIMO): tabellarische Feature-Matrizen.  
  - **Native Multi-Horizon DL** (z. B. TFT, N-HiTS): sequenzielle Eingaben inkl. known-future-Features; explizite Lags/Rollings je nach Architektur nicht zwingend erforderlich.

