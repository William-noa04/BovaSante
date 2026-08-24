"""
Moteur d'inférence pour l'outil `predict_disease` du chatbot.

Fonctionnement :
- Le modèle chargé est le modèle de fusion multimodale complet (best_model.pt).
- Comme le chatbot ne dispose ni d'image ni de texte symptomatique structuré
  au moment de l'appel, l'image et le texte sont mis à zéro (torch.zeros),
  et seule la branche tabulaire contribue réellement à la prédiction.
  C'est exactement le mode "tabular_only" déjà utilisé lors de l'ablation
  (voir Étape 3 du notebook), dont l'accuracy mesurée est d'environ 23 %.
  => Le résultat doit toujours être présenté à l'éleveur comme une indication
     à faire confirmer par un vétérinaire, jamais comme un diagnostic certain.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from model_def import MultimodalFusionModel

# ----------------------------------------------------------------------
# Configuration des chemins — ADAPTE ENCODERS_DIR à ton environnement
# ----------------------------------------------------------------------
ENCODERS_DIR = os.environ.get(
    "ENCODERS_DIR",
    r"C:\Users\user\Documents\VENV\ProjetSoutenance\data_processed\encoders",
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    r"C:\Users\user\Documents\VENV\ProjetSoutenance\data_processed",
)

MAX_SYMPTOM_LEN = 30
CANONICAL_CLASSES = ["healthy", "lumpy_skin_disease", "foot_and_mouth_disease"]
CLASS_LABELS_FR = {
    "healthy": "Sain",
    "lumpy_skin_disease": "Dermatose Nodulaire Contagieuse Bovine (DNCB)",
    "foot_and_mouth_disease": "Fièvre Aphteuse (FA)",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Champs que l'éleveur peut raisonnablement fournir en conversation.
# Tout ce qui n'est pas dans cette liste est complété avec tabular_defaults.json
FARMER_NUMERIC_FIELDS = [
    "Age_Months", "Weight_kg", "Body_Temperature_C", "Heart_Rate_bpm",
    "Respiratory_Rate", "Body_Condition_Score", "Milk_Yield_L",
    "Previous_Week_Avg_Yield",
]
FARMER_CATEGORICAL_FIELDS = ["FMD_Vaccine"]


class InferenceEngine:
    """Charge tous les artefacts une seule fois au démarrage du serveur."""

    def __init__(self):
        missing = []

        def _load(name, loader):
            path = os.path.join(ENCODERS_DIR, name)
            if not os.path.exists(path):
                missing.append(path)
                return None
            return loader(path)

        self.schema = _load("tabular_schema.json", lambda p: json.load(open(p, encoding="utf-8")))
        self.defaults = _load("tabular_defaults.json", lambda p: json.load(open(p, encoding="utf-8")))
        self.scaler = _load("tabular_scaler.pkl", joblib.load)
        self.onehot = _load("tabular_onehot_encoder.pkl", joblib.load)
        self.label_encoders = _load("tabular_label_encoders.pkl", joblib.load) or {}
        self.target_encoder = _load("tabular_target_encoder.pkl", joblib.load)
        vocab_path = os.path.join(ENCODERS_DIR, "symptoms_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                self.vocab_size = len(json.load(f))
        else:
            missing.append(vocab_path)
            self.vocab_size = 2  # PAD + UNK minimum, modèle non fonctionnel de toute façon

        if missing:
            raise FileNotFoundError(
                "Fichiers d'artefacts manquants, le serveur ne peut pas démarrer:\n"
                + "\n".join(missing)
                + "\n\nAs-tu bien exécuté compute_defaults.py et copié les fichiers "
                  "générés par le notebook de prétraitement dans ENCODERS_DIR ?"
            )

        tabular_input_dim = len(self.schema["feature_columns_order"])
        self.model = MultimodalFusionModel(
            vocab_size=self.vocab_size,
            tabular_input_dim=tabular_input_dim,
            num_classes=len(CANONICAL_CLASSES),
        ).to(DEVICE)

        checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Modèle introuvable: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _build_tabular_row(self, farmer_inputs: dict) -> pd.DataFrame:
        """Construit une ligne brute complète (1 ligne) = defaults + overrides éleveur."""
        row = dict(self.defaults)  # copie
        for key, value in farmer_inputs.items():
            if value is not None and key in row:
                row[key] = value
        return pd.DataFrame([row])

    def _encode_row(self, df_row: pd.DataFrame) -> np.ndarray:
        """Reproduit exactement le pipeline d'encodage du notebook de prétraitement."""
        num_cols = self.schema["numeric_cols"]
        low_card_cols = self.schema["low_card_cols"]
        high_card_cols = self.schema["high_card_cols"]
        feature_order = self.schema["feature_columns_order"]

        parts = [df_row[num_cols].reset_index(drop=True)]

        if low_card_cols and self.onehot is not None:
            ohe_array = self.onehot.transform(df_row[low_card_cols])
            ohe_df = pd.DataFrame(
                ohe_array, columns=self.onehot.get_feature_names_out(low_card_cols)
            )
            parts.append(ohe_df)

        for col in high_card_cols:
            le = self.label_encoders.get(col)
            raw_val = df_row[col].iloc[0]
            try:
                encoded = le.transform([raw_val])[0] if le is not None else 0
            except ValueError:
                # Valeur jamais vue à l'entraînement -> on retombe sur la classe la plus fréquente
                encoded = 0
            parts.append(pd.DataFrame({f"{col}_encoded": [encoded]}))

        df_encoded = pd.concat(parts, axis=1)
        df_encoded = df_encoded.reindex(columns=feature_order, fill_value=0)

        # Normalisation des colonnes numériques avec le scaler entraîné
        present_num_cols = [c for c in num_cols if c in df_encoded.columns]
        df_encoded[present_num_cols] = self.scaler.transform(df_encoded[present_num_cols])

        return df_encoded.to_numpy(dtype=np.float32)

    def predict(self, farmer_inputs: dict) -> dict:
        df_row = self._build_tabular_row(farmer_inputs)
        tabular_vector = self._encode_row(df_row)

        tabular_tensor = torch.tensor(tabular_vector, dtype=torch.float32).to(DEVICE)
        dummy_image = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(DEVICE)
        dummy_text = torch.zeros((1, MAX_SYMPTOM_LEN), dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            logits = self.model(dummy_image, dummy_text, tabular_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CANONICAL_CLASSES[pred_idx]

        return {
            "predicted_class": pred_class,
            "predicted_class_fr": CLASS_LABELS_FR[pred_class],
            "confidence": round(float(probs[pred_idx]), 3),
            "probabilities": {
                CLASS_LABELS_FR[c]: round(float(p), 3)
                for c, p in zip(CANONICAL_CLASSES, probs)
            },
            "reliability_note": (
                "Cette estimation se base uniquement sur les paramètres physiologiques "
                "fournis (branche tabulaire du modèle, fiabilité mesurée ~23% en isolation "
                "lors des tests). Elle doit être confirmée par un examen vétérinaire, "
                "idéalement complété par une photo des lésions."
            ),
        }


# Instance chargée une seule fois au démarrage de l'API (voir main.py)
engine: "InferenceEngine | None" = None


def get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine
