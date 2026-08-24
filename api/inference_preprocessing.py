"""
inference_preprocessing.py
=============================
Reproduit, pour UNE SEULE observation reçue par l'API, exactement le même
pipeline de prétraitement que celui utilisé à l'entraînement (Étape 1),
en réutilisant les encodeurs/scaler sauvegardés (.pkl) et le schéma de
colonnes (tabular_schema.json) plutôt qu'en recalculant quoi que ce soit
sur une seule ligne (ce qui serait statistiquement incorrect : on ne peut
pas déduire une cardinalité ou une moyenne à partir d'un seul exemple).
"""

import io
import json
import os

import joblib
import numpy as np
import torch
from PIL import Image

IMG_SIZE = (224, 224)

# Indices ImageNet-1k (ResNet18_Weights.IMAGENET1K_V1.meta["categories"]) les plus
# proches d'un bovin : le jeu de 1000 classes n'a pas de "cow"/"cattle" dédiée.
BOVINE_IMAGENET_CLASSES = {345: "ox", 346: "water buffalo", 347: "bison"}
BOVINE_GATE_THRESHOLD = 0.15


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convertit une image PIL déjà décodée en tenseur (1, 3, 224, 224).

    IMPORTANT : reproduit EXACTEMENT le prétraitement utilisé dans
    MultimodalCattleDataset.__getitem__ à l'entraînement — redimensionnement
    224x224 puis division simple par 255, SANS normalisation ImageNet
    (mean/std). Le modèle n'a jamais vu d'images normalisées de cette façon ;
    appliquer une normalisation différente à l'inférence dégraderait les
    prédictions de façon silencieuse (pas d'erreur, juste de moins bons résultats).
    """
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(image_array)
    return tensor.unsqueeze(0)


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Raccourci decode + preprocess, pour les appelants qui n'ont pas besoin de l'image PIL intermédiaire."""
    return preprocess_image(decode_image_bytes(image_bytes))


def is_bovine_image(image: Image.Image, gate_model, gate_transform, device) -> tuple[bool, float]:
    """
    Garde-fou "image de bovin uniquement", basé sur un ResNet18 pré-entraîné
    ImageNet (indépendant du modèle de fusion). Le score est la masse de
    probabilité cumulée sur les 3 classes bovines du jeu ImageNet-1k.

    Limite connue et acceptée : le dataset d'entraînement du projet contient
    surtout des gros plans (muselières, lésions) qu'un classifieur ImageNet
    générique ne peut pas rattacher à "ox"/"water buffalo"/"bison" — une partie
    de ces photos légitimes sera donc rejetée à tort par ce filtre strict.
    """
    x = gate_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(gate_model(x), dim=1)[0]
    score = sum(probs[i].item() for i in BOVINE_IMAGENET_CLASSES)
    return score >= BOVINE_GATE_THRESHOLD, score


def preprocess_symptoms_text(text: str, vocab, max_len: int) -> torch.Tensor:
    """Nettoie et encode une phrase de symptômes en tenseur (1, max_len)."""
    import re
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-zA-Z0-9àâäéèêëïîôöùûüç,\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    ids = vocab.encode(cleaned, max_len=max_len)
    return torch.tensor([ids], dtype=torch.long)


class TabularPreprocessor:
    """
    Charge une seule fois les encodeurs/scaler/schéma sauvegardés, puis
    transforme un dict de données tabulaires brutes en tenseur prêt pour
    le modèle, dans le MÊME ordre de colonnes qu'à l'entraînement.
    """

    def __init__(self, encoders_dir: str):
        self.scaler = joblib.load(os.path.join(encoders_dir, "tabular_scaler.pkl"))

        ohe_path = os.path.join(encoders_dir, "tabular_onehot_encoder.pkl")
        self.onehot_encoder = joblib.load(ohe_path) if os.path.exists(ohe_path) else None

        le_path = os.path.join(encoders_dir, "tabular_label_encoders.pkl")
        self.label_encoders = joblib.load(le_path) if os.path.exists(le_path) else {}

        with open(os.path.join(encoders_dir, "tabular_schema.json"), "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def transform(self, raw_data: dict) -> torch.Tensor:
        import pandas as pd

        df = pd.DataFrame([raw_data])

        numeric_cols = self.schema["numeric_cols"]
        low_card_cols = self.schema["low_card_cols"]
        high_card_cols = self.schema["high_card_cols"]
        feature_order = self.schema["feature_columns_order"]

        parts = [df.drop(columns=low_card_cols + high_card_cols, errors="ignore")]

        # IMPORTANT : à l'entraînement, TOUTES les colonnes catégorielles (y compris
        # les indicateurs de vaccination 0/1) ont été converties en texte avant
        # l'encodage (`.astype(str).str.strip()`). Si on ne fait pas la même chose
        # ici, un entier 0 ne correspondra jamais à la chaîne "0" apprise par
        # l'encodeur, et handle_unknown="ignore" produira silencieusement un
        # vecteur entièrement à zéro — une erreur silencieuse, la pire espèce.
        for c in low_card_cols + high_card_cols:
            df[c] = df[c].astype(str).str.strip()

        if low_card_cols and self.onehot_encoder is not None:
            ohe_array = self.onehot_encoder.transform(df[low_card_cols])
            ohe_df = pd.DataFrame(
                ohe_array,
                columns=self.onehot_encoder.get_feature_names_out(low_card_cols),
                index=df.index,
            )
            parts.append(ohe_df)

        for c in high_card_cols:
            le = self.label_encoders.get(c)
            raw_value = str(df.at[0, c])
            if le is not None and raw_value in le.classes_:
                encoded_value = le.transform([raw_value])[0]
            else:
                # Valeur jamais vue à l'entraînement (ex: nouvelle race/pays) :
                # on retombe sur la première classe connue plutôt que de planter.
                encoded_value = 0
            parts.append(pd.DataFrame({f"{c}_encoded": [encoded_value]}))

        df_encoded = pd.concat(parts, axis=1)
        df_encoded[numeric_cols] = self.scaler.transform(df_encoded[numeric_cols])

        # Réordonne / complète EXACTEMENT selon le schéma d'entraînement
        df_encoded = df_encoded.reindex(columns=feature_order, fill_value=0.0)

        values = df_encoded.to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)
