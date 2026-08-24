"""
À exécuter UNE SEULE FOIS, en local, sur ta machine Windows, dans le même
environnement que ton notebook de prétraitement (Étape 1).

Ce script calcule, pour chaque variable brute du dataset tabulaire, une valeur
par défaut (médiane pour les numériques, mode pour les catégorielles) qui sera
utilisée par le chatbot pour compléter les champs que l'éleveur ne fournit pas
en conversation.

Sortie : data_processed/encoders/tabular_defaults.json

Usage :
    python compute_defaults.py
(adapte BASE_DIR si besoin, doit correspondre à ton notebook de prétraitement)
"""
import os
import json
import pandas as pd

BASE_DIR = r"C:\Users\user\Documents\VENV\ProjetSoutenance"
TABULAR_CSV = os.path.join(BASE_DIR, "global_cattle_disease_detection_dataset.csv")
ENCODERS_DIR = os.path.join(BASE_DIR, "data_processed", "encoders")

TABULAR_CATEGORICAL_COLS = [
    "Breed", "Region", "Country", "Climate_Zone", "Management_System",
    "Lactation_Stage", "Feed_Type", "Season",
    "FMD_Vaccine", "Brucellosis_Vaccine", "HS_Vaccine", "BQ_Vaccine",
    "Anthrax_Vaccine", "IBR_Vaccine", "BVD_Vaccine", "Rabies_Vaccine",
]

TABULAR_NUMERIC_COLS = [
    "Age_Months", "Weight_kg", "Parity", "Days_in_Milk", "Feed_Quantity_kg",
    "Water_Intake_L", "Walking_Distance_km", "Grazing_Duration_hrs",
    "Rumination_Time_hrs", "Resting_Hours", "Body_Temperature_C",
    "Heart_Rate_bpm", "Respiratory_Rate", "Ambient_Temperature_C",
    "Humidity_percent", "Housing_Score", "Milk_Yield_L",
    "Previous_Week_Avg_Yield", "Body_Condition_Score", "Milking_Interval_hrs",
]


def main():
    df = pd.read_csv(TABULAR_CSV)
    df = df.drop_duplicates()

    defaults = {}

    for col in TABULAR_NUMERIC_COLS:
        if col in df.columns:
            defaults[col] = float(df[col].median())
        else:
            print(f"[WARN] colonne numérique absente: {col}")

    for col in TABULAR_CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            defaults[col] = str(mode_val.iloc[0]) if not mode_val.empty else "unknown"
        else:
            print(f"[WARN] colonne catégorielle absente: {col}")

    os.makedirs(ENCODERS_DIR, exist_ok=True)
    out_path = os.path.join(ENCODERS_DIR, "tabular_defaults.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, ensure_ascii=False, indent=2)

    print(f"✅ Valeurs par défaut sauvegardées: {out_path}")
    print(json.dumps(defaults, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
