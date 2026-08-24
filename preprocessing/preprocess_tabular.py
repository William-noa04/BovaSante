"""
preprocess_tabular.py
=======================
- Supprime les colonnes identifiants (Cattle_ID, Farm_ID, Date).
- Encode les variables catégorielles (OneHot ou Label selon cardinalité).
- Normalise les variables numériques (StandardScaler).
- Encode la cible Disease_Status.
- Sauvegarde le CSV final + les encodeurs/scaler.
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    TABULAR_CSV, OUTPUT_DIR, ENCODERS_DIR,
    TABULAR_TARGET_COL, TABULAR_CATEGORICAL_COLS,
    TABULAR_NUMERIC_COLS, TABULAR_DROP_COLS,
)

# Seuil : au-delà de ce nombre de modalités, on utilise LabelEncoder plutôt que OneHot
ONEHOT_MAX_CARDINALITY = 10


def process_tabular():
    df = pd.read_csv(TABULAR_CSV)

    # 1. Nettoyage basique
    df = df.drop_duplicates()
    df = df.drop(columns=[c for c in TABULAR_DROP_COLS if c in df.columns], errors="ignore")

    # Vérifie la présence des colonnes attendues
    missing_cat = [c for c in TABULAR_CATEGORICAL_COLS if c not in df.columns]
    missing_num = [c for c in TABULAR_NUMERIC_COLS if c not in df.columns]
    if missing_cat:
        print(f"[WARN] Colonnes catégorielles absentes (ignorées): {missing_cat}")
    if missing_num:
        print(f"[WARN] Colonnes numériques absentes (ignorées): {missing_num}")

    cat_cols = [c for c in TABULAR_CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in TABULAR_NUMERIC_COLS if c in df.columns]

    # Imputation simple des valeurs manquantes
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str).str.strip()

    # 2. Encodage de la cible
    target_encoder = LabelEncoder()
    df[TABULAR_TARGET_COL] = df[TABULAR_TARGET_COL].astype(str).str.strip().str.lower()
    df["target_encoded"] = target_encoder.fit_transform(df[TABULAR_TARGET_COL])
    joblib.dump(target_encoder, os.path.join(ENCODERS_DIR, "tabular_target_encoder.pkl"))

    # 3. Encodage des variables catégorielles
    low_card_cols = [c for c in cat_cols if df[c].nunique() <= ONEHOT_MAX_CARDINALITY]
    high_card_cols = [c for c in cat_cols if c not in low_card_cols]

    # On garde une copie texte brute de la cible (utile pour le mapping de
    # classes canoniques à l'étape 2), en plus de la version encodée.
    raw_target = df[[TABULAR_TARGET_COL]].copy()

    encoded_parts = [df.drop(columns=cat_cols + [TABULAR_TARGET_COL]), raw_target]

    if low_card_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_array = ohe.fit_transform(df[low_card_cols])
        ohe_df = pd.DataFrame(
            ohe_array, columns=ohe.get_feature_names_out(low_card_cols), index=df.index
        )
        encoded_parts.append(ohe_df)
        joblib.dump(ohe, os.path.join(ENCODERS_DIR, "tabular_onehot_encoder.pkl"))

    label_encoders = {}
    for c in high_card_cols:
        le = LabelEncoder()
        df[c + "_encoded"] = le.fit_transform(df[c])
        label_encoders[c] = le
        encoded_parts.append(df[[c + "_encoded"]])
    if label_encoders:
        joblib.dump(label_encoders, os.path.join(ENCODERS_DIR, "tabular_label_encoders.pkl"))

    df_encoded = pd.concat(encoded_parts, axis=1)

    # 4. Normalisation des variables numériques
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    joblib.dump(scaler, os.path.join(ENCODERS_DIR, "tabular_scaler.pkl"))

    # 5. Sauvegarde
    out_csv = os.path.join(OUTPUT_DIR, "tabular_clean.csv")
    df_encoded.to_csv(out_csv, index=False)

    # Sauvegarde du schéma exact (colonnes, groupement, ordre final) pour que
    # l'API d'inférence puisse reconstruire un vecteur tabulaire identique à
    # celui utilisé à l'entraînement, sans avoir à recalculer la cardinalité
    # (impossible à faire correctement sur une seule ligne en production).
    feature_columns = [c for c in df_encoded.columns if c not in ("Disease_Status", "target_encoded")]
    schema = {
        "numeric_cols": num_cols,
        "low_card_cols": low_card_cols,
        "high_card_cols": high_card_cols,
        "feature_columns_order": feature_columns,
    }
    with open(os.path.join(ENCODERS_DIR, "tabular_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"✅ Tabulaire nettoyé: {df_encoded.shape[0]} lignes, {df_encoded.shape[1]} colonnes -> {out_csv}")
    print("Classes cible:", list(target_encoder.classes_))
    return df_encoded


if __name__ == "__main__":
    process_tabular()
