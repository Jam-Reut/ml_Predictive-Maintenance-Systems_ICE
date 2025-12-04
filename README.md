🚄 Predictive Maintenance – Machine Learning Projekt (ICE-Komponenten)
Dieses Projekt demonstriert ein vollständiges Predictive-Maintenance-System für ICE-Zugkomponenten, basierend auf einem "realistisch" generierten synthetischen Sensor-Datensatz:
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jam-Reut/ml_Predictive-Maintenance-Systems_ICE/HEAD?labpath=predictive_maintenance_exam.ipynb)

## Ziel dieses Projekts:
Vorhersage, ob innerhalb der nächsten 30 Tage ein technischer Ausfall auftritt (failure_within_30d)
Aufbau eines vollständigen ML-Workflows für Prüfungs-, Forschungs- und Entwicklungszwecke.

## Projektstruktur:
📦 predictive-maintenance-ice
   - data_generator_818370.py
   - predictive_maintenance_exam_v2.csv
   - predictive_maintenance_exam_notebook_818370.ipynb
   - README.md

## Data_generator.py
Erzeugt einen physikalisch plausiblen Predictive-Maintenance-Datensatz, der reale Zusammenhänge zwischen Zugkomponenten simuliert.
Enthält u. a.:
Temperatur- und Vibrationsmessungen
mechanische Belastungsfaktoren
Wartungsintervalle und Fehlerhistorien
Umweltbedingungen (hot, cold, wet, normal)
erweiterte Features wie
temperature_diff
stress_index (kombiniert mehrere Risikofaktoren)

Das Failure-Label wird mit einer Sigmoid-basierten Risikofunktion erzeugt, sodass ML-Modelle realistische Muster lernen können.
Die resultierende Failure-Rate beträgt ca. 10 %, ideal für Klassifikationsmodelle

## predictive_maintenance_exam.csv
Fertiger Datensatz mit ca. 6000 Instanzen und:
12 numerischen Sensor-Features
3 kategorialen System-Attributen (train_line, shift, environment_mode)
Zielvariable failure_within_30d
Der Datensatz ist sauber, konsistent, keine NaN-Werte,
und enthält deutliche Muster, die für ML lernbar sind.

## predictive_maintenance_exam_notebook.ipynb
Das zentrale Notebook führt den vollständigen Workflow durch:
1. Explorative Datenanalyse (EDA)
Histogramme aller Features
Korrelationsmatrix in hellen Farben für beste Lesbarkeit
Scatterplots & Pairplots
Jointplots (Temperatur vs. Vibration)
2. Datenvorbereitung
Train/Test-Split (stratified)
ColumnTransformer:
StandardScaler für numerische Features
OneHotEncoder für kategoriale Features
scikit-learn Pipeline für saubere Reproduzierbarkeit
3. Modelle
Logistische Regression
Random Forest
Gradient Boosting
Alle Modelle werden vollständig evaluiert.
4. Modell-Evaluierung
Klassifikationsbericht
Confusion-Matrizen (helle Farbpalette)
ROC-Kurven
ROC-AUC
Optimaler Threshold nach Youden-Index
5. Modellvergleich
Überlagerte ROC-Kurve aller Modelle.
6. Abschlussmeldung
Nach vollständiger Ausführung:
„Alle Schritte wurden erfolgreich ausgeführt – gut gemacht!“

## 🧾 Ergebnis beim Ausführen
Nach Ausführung des Notebooks erhältst du:
vollständige EDA
robuste Datenaufbereitung
mehrere ML-Modelle im Vergleich
gut lesbare Heatmaps & Diagramme
optimal gewählten Threshold
klare Interpretation
finale Erfolgsmeldung

