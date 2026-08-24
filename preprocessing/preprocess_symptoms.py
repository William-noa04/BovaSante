"""
preprocess_symptoms.py
========================
Colonnes attendues: image, disease, symptoms, severity

- Nettoie le texte des symptômes (minuscules, ponctuation, espaces).
- Encode 'disease' (LabelEncoder) et 'severity' si catégoriel (ex: mild/moderate/severe).
- Sauvegarde le CSV nettoyé + les encodeurs (pour réutilisation en inférence).
"""

import os
import re
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import SYMPTOMS_CSV, OUTPUT_DIR, ENCODERS_DIR


def clean_text(text: str) -> str:
    """Nettoyage basique du texte des symptômes."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9àâäéèêëïîôöùûüç,\s]", " ", text)  # garde lettres/chiffres/virgules
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_symptoms():
    df = pd.read_csv(SYMPTOMS_CSV)

    required_cols = {"image", "disease", "symptoms", "severity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {SYMPTOMS_CSV}: {missing}")

    # 1. Nettoyage
    df["symptoms_clean"] = df["symptoms"].apply(clean_text)
    df["disease"] = df["disease"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates()
    df = df.dropna(subset=["symptoms_clean", "disease"])
    df = df[df["symptoms_clean"] != ""]

    # 2. Encodage de la cible 'disease'
    disease_encoder = LabelEncoder()
    df["disease_encoded"] = disease_encoder.fit_transform(df["disease"])

    # 3. Encodage de 'severity' si catégoriel (ex: mild/moderate/severe)
    severity_encoder = None
    if df["severity"].dtype == object:
        severity_encoder = LabelEncoder()
        df["severity_encoded"] = severity_encoder.fit_transform(
            df["severity"].astype(str).str.strip().str.lower()
        )
    else:
        df["severity_encoded"] = df["severity"]

    # 4. Sauvegarde
    out_csv = os.path.join(OUTPUT_DIR, "symptoms_clean.csv")
    df.to_csv(out_csv, index=False)

    joblib.dump(disease_encoder, os.path.join(ENCODERS_DIR, "disease_label_encoder.pkl"))
    if severity_encoder is not None:
        joblib.dump(severity_encoder, os.path.join(ENCODERS_DIR, "severity_label_encoder.pkl"))

    print(f"✅ Symptômes nettoyés: {len(df)} lignes -> {out_csv}")
    print("Classes de maladie:", list(disease_encoder.classes_))
    return df


if __name__ == "__main__":
    process_symptoms()
