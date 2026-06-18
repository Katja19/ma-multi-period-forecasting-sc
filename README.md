# Masterarbeit - Mehrperiodisches Forecasting für dynamische Entscheidungsfindung 
Autor: Katja Gegg  
Universität: Julius-Maximilian Universität Würzburg

## Dokumentation
Eine genauere Dokumentation findet sich hier: https://katja19.github.io/ma-multi-period-forecasting-sc/

## Setup-Anleitung (im VSC-Terminalfenster)
1. Stelle sicher, dass du dich im Projektverzeichnis befindest.
2. Erstelle eine virtuelle Umgebung `.venv`:
   ```sh
   python -m venv .venv
   ```
3. Aktiviere die virtuelle Umgebung:
   ```sh
   .venv\Scripts\activate
   ```
4. Installiere die benötigten Abhängigkeiten:
   ```sh
   pip install -r requirements.txt
   ```
5. Extra: Du kannst die virtuelle Umgebung mit folgendem Befehl deaktivieren:
   ```sh
   deactivate
   ```

## Zugriff auf Datensätze
Die CSV-Dateien der Datensätze wurden wegen ihrer größe als Zip-Datein hochgeladen und müssen erst enpackt werden.

## Python-Version
Dieses Projekt wurde mit Python 3.11.9 und Pip 24.0 erstellt. Du kannst deine Python-Version mit folgendem Befehl überprüfen:
   ```sh
   python --version
   ```

## Lokale Nutzung des mkDocs-Servers in VSC
Die Dokumentation wurde mit Material for MkDocs erstellt: https://squidfunk.github.io/mkdocs-material/

### Starten des mkDocs-Servers
1. Öffne das Terminal in Visual Studio Code.
2. Navigiere zum Projektverzeichnis:
   ```sh
   cd pfad/zu/deinem/projektverzeichnis/multi-period-forecasting
   ```
   Ersetze `pfad/zu/deinem/projektverzeichnis` durch den tatsächlichen Pfad, in dem du das Projekt gespeichert hast.
   <!--z.B. `/c:/Users/User/Code/VSCProjects/Master_Thesis/multi-period-forecasting`-->
3. Installiere mkDocs, falls noch nicht geschehen (sollte installiert sein, wenn du die Abhängigkeiten installiert hast):
   ```sh
   pip install mkdocs
   ```
4. Starte den mkDocs-Server:
   ```sh
   mkdocs serve
   ```
5. Öffne deinen Webbrowser und gehe zu `http://127.0.0.1:8000`, um die Dokumentation anzusehen.

### Stoppen des mkDocs-Servers
1. Gehe in das Terminal, in dem der mkDocs-Server läuft.
2. Drücke `Strg + C`, um den Server zu stoppen.

## Hinweis
Die Datei `safe_start.ipynb` dient als Ausführungspunkt, um eine Pipeline zu starten, mit der ein Modell trainiert und Vorhersagen gemacht werden können. In diesem Notebook kann der Nutzer verschiedene Übergabeparameter wählen, um beispielsweise das gewünschte Modell auszuwählen oder auch welcher Datensatz verwendet werden soll und mehr (führ Details sieh die Informationen in der genannten Datei.)