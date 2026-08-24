"""
model_loader.py
=================
Charge une seule fois, au démarrage de l'API, les deux modèles entraînés
ainsi que tous les artefacts nécessaires à l'inférence (vocabulaire,
encodeurs tabulaires). Évite de recharger ces objets à chaque requête.
"""

import json
import os
import sys

import torch
from torchvision.models import resnet18, ResNet18_Weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    OUTPUT_DIR, ENCODERS_DIR, CANONICAL_CLASSES, NUM_CLASSES, MAX_SYMPTOM_LEN,
)
from data_utils.vocab import Vocab
from models.fusion_model import MultimodalFusionModel
from models.tabular_only_model import TabularOnlyClassifier
from api.inference_preprocessing import TabularPreprocessor

import joblib


class ModelRegistry:
    """Conteneur unique pour tous les modèles et artefacts chargés en mémoire."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = None
        self.multimodal_model = None
        self.multimodal_classes = CANONICAL_CLASSES

        self.tabular_only_model = None
        self.tabular_only_classes = None

        self.tabular_preprocessor = None
        self.tabular_defaults = None

        self.bovine_gate_model = None
        self.bovine_gate_transform = None

    def load_all(self):
        # --- Vocabulaire des symptômes ---
        vocab_path = os.path.join(ENCODERS_DIR, "symptoms_vocab.json")
        self.vocab = Vocab.load(vocab_path)

        # --- Préprocesseur tabulaire (encodeurs + schéma de colonnes) ---
        self.tabular_preprocessor = TabularPreprocessor(ENCODERS_DIR)

        # --- Valeurs par défaut (médianes/modes) pour /predict/simplified ---
        defaults_path = os.path.join(ENCODERS_DIR, "tabular_defaults.json")
        if os.path.exists(defaults_path):
            with open(defaults_path, encoding="utf-8") as f:
                self.tabular_defaults = json.load(f)
        else:
            print(f"[WARN] Valeurs par défaut introuvables: {defaults_path} (/predict/simplified sera indisponible)")

        # --- Garde-fou "image de bovin uniquement" (ResNet18 ImageNet, indépendant du modèle de fusion) ---
        gate_weights = ResNet18_Weights.IMAGENET1K_V1
        self.bovine_gate_model = resnet18(weights=gate_weights).to(self.device)
        self.bovine_gate_model.eval()
        self.bovine_gate_transform = gate_weights.transforms()

        # --- Modèle multimodal ---
        multimodal_checkpoint = os.path.join(OUTPUT_DIR, "best_model.pt")
        if os.path.exists(multimodal_checkpoint):
            tabular_input_dim = len(self.tabular_preprocessor.schema["feature_columns_order"])
            self.multimodal_model = MultimodalFusionModel(
                vocab_size=len(self.vocab),
                tabular_input_dim=tabular_input_dim,
                num_classes=NUM_CLASSES,
            ).to(self.device)
            self.multimodal_model.load_state_dict(
                torch.load(multimodal_checkpoint, map_location=self.device)
            )
            self.multimodal_model.eval()
        else:
            print(f"[WARN] Checkpoint multimodal introuvable: {multimodal_checkpoint}")

        # --- Modèle tabulaire seul ---
        tabular_only_checkpoint = os.path.join(OUTPUT_DIR, "best_tabular_only_model.pt")
        target_encoder_path = os.path.join(ENCODERS_DIR, "tabular_target_encoder.pkl")
        if os.path.exists(tabular_only_checkpoint) and os.path.exists(target_encoder_path):
            target_encoder = joblib.load(target_encoder_path)
            self.tabular_only_classes = list(target_encoder.classes_)
            tabular_input_dim = len(self.tabular_preprocessor.schema["feature_columns_order"])
            self.tabular_only_model = TabularOnlyClassifier(
                input_dim=tabular_input_dim,
                num_classes=len(self.tabular_only_classes),
            ).to(self.device)
            self.tabular_only_model.load_state_dict(
                torch.load(tabular_only_checkpoint, map_location=self.device)
            )
            self.tabular_only_model.eval()
        else:
            print(f"[WARN] Checkpoint tabulaire seul introuvable: {tabular_only_checkpoint}")


# Instance unique partagée par toute l'application (pattern singleton simple)
registry = ModelRegistry()
