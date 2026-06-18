---
title: Getting Started
author: Katja Gegg
university: Julius-Maximilian Universität Würzburg
---
# Getting started

## Einrichtung der virtuellen Umgebung

1. Stelle sicher, dass du dich im Projekt-Stammverzeichnis im Terminal befindest.

    ```sh
    cd pfad\zu\deinem\projekt\multi-period-forecasting
    ```

2. Erstelle eine virtuelle Umgebung namens `.venv`:

    ```sh
    python -m venv .venv
    ```

3. Aktiviere die virtuelle Umgebung:

    === "Windows"
        ```sh
        .venv\Scripts\activate
        ```

    === "macOS / Linux"
        ```sh
        source .venv/bin/activate
        ```

4. Installiere die benötigten Abhängigkeiten:

    ```sh
    pip install -r requirements.txt
    ```

5. (Optional) Deaktiviere die virtuelle Umgebung:

    ```sh
    deactivate
    ```

## Zugriff auf Datensätze

Die in dieser Arbeit verwendeten Datensätze wurden in ZIP-Format in ihren angedachten Ordener hochgeladen und müssen vor benutzun entpackt und richtig gespeichert werden (vgl. Projektstuktur weiter unten).

## Python-Version

Dieses Projekt wurde entwickelt mit:

- Python 3.11.9  
- Pip 24.0

Überprüfe deine aktuelle Python-Version mit:
```sh
python --version
```

## Dokumentation (MkDocs)
Die Dokumentation wird mit [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) erstellt und über GitHub Pages veröffentlicht. Änderungen an der Dokumentation im `docs/`-Verzeichnis werden nach dem Pushen automatisch auf der Projekt-Webseite aktualisiert. Alternativ kann die Dokumentation auch lokal über einen Host betrachtet werden, indem der MkDocs-Server wie folgend beschrieben gestartet wird.

### MkDocs-Server starten

1. Öffne das Terminal in deiner IDE, z. B. Visual Studio Code.
2. Navigiere ins Projektverzeichnis:
   ```sh
   cd pfad\zu\deinem\projekt\multi-period-forecasting
   ```
   Ersetze den Platzhalter durch deinen tatsächlichen Dateipfad.
3. Installiere MkDocs (falls noch nicht installiert):
   ```sh
   pip install mkdocs
   ```
4. Starte den MkDocs-Server:
   ```sh
   mkdocs serve
   ```
5. Öffne lokalen Host in deinem Browser:
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

### MkDocs-Server stoppen

Drücke `Strg + C` im Terminal, um den Server zu stoppen.


### Projektstruktur

??? info "Projektstruktur anzeigen"
    ```text
    root/
    ├─ data/
    │  ├─ basic_preprocessed/
    │  │  ├─ bakery_basic_prepro.csv
    │  │  ├─ basic_preprocessed.zip
    │  │  └─ m5_basic_prepro.csv
    │  └─ raw/
    │     ├─ dataBakery.csv
    │     ├─ dataM5.csv
    │     └─ raw.zip
    ├─ docs/
    ├─ outputs/
    │  ├─ eda/
    │  │  ├─ bakery/seasonality_flags_bakery_traincut_val28_test28.csv
    │  │  └─ m5/seasonality_flags_m5_traincut_val28_test28.csv
    │  ├─ evaluation/
    │  │  ├─ agg_by_class.csv
    │  │  ├─ agg_by_model_wrapper.csv
    │  │  ├─ agg_by_model_wrapper_dataset.csv
    │  │  ├─ best_per_dataset.csv
    │  │  └─ runs_summary.csv
    │  ├─ dl_val_summaries.json
    │  └─ hptuned_params.json
    ├─ src/
    │  ├─ 0_eda_prepro/
    │  │  ├─ create_images_for_docs.ipynb
    │  │  ├─ eda_prepro_backery_v2.ipynb
    │  │  └─ eda_prepro_m5_v2.ipynb
    │  ├─ 1_modelling/
    │  │  ├─ checkpoints/
    │  │  ├─ config.py
    │  │  ├─ core.py
    │  │  ├─ datasets.py
    │  │  ├─ hpo.py
    │  │  ├─ model_utils.py
    │  │  ├─ models_dl.py
    │  │  ├─ models_lgbm.py
    │  │  ├─ models_naive.py
    │  │  ├─ models_xgb.py
    │  │  ├─ pipeline.py
    │  │  ├─ results_logger.py
    │  │  ├─ safe_runner.py
    │  │  ├─ safe_start.ipynb
    │  │  └─ wandb_utils.py
    │  └─ 2_evaluation/
    │     └─ analysis_notebook.ipynb
    ├─ .gitignore
    ├─ mkdocs.yml
    ├─ README.md
    ├─ requirements.txt
    └─ tree.txt
    ```