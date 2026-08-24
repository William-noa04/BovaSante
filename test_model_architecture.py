"""
test_model_architecture.py
=============================
Vérifie que le modèle multimodal complet fonctionne (forward pass) avec
des tenseurs factices, AVANT de brancher les vraies données. Utile pour
détecter les erreurs de dimensions sans attendre l'Étape 3.

Usage:
    python test_model_architecture.py
"""

import torch
from models.fusion_model import MultimodalFusionModel
from config import (
    NUM_CLASSES, MAX_SYMPTOM_LEN, TEXT_EMBED_DIM, TEXT_HIDDEN_DIM,
    TEXT_OUTPUT_DIM, IMAGE_OUTPUT_DIM, TABULAR_HIDDEN_DIMS,
    TABULAR_OUTPUT_DIM, FUSION_HIDDEN_DIM, DROPOUT,
)

# Valeurs factices — remplace VOCAB_SIZE et TABULAR_INPUT_DIM par les
# vraies valeurs une fois le vocabulaire et le tabulaire encodé générés
# (Étape 1) : len(vocab) et dataset.tabular_input_dim
FAKE_VOCAB_SIZE = 500
FAKE_TABULAR_INPUT_DIM = 40
FAKE_BATCH_SIZE = 8


def main():
    model = MultimodalFusionModel(
        vocab_size=FAKE_VOCAB_SIZE,
        tabular_input_dim=FAKE_TABULAR_INPUT_DIM,
        num_classes=NUM_CLASSES,
        image_output_dim=IMAGE_OUTPUT_DIM,
        text_embed_dim=TEXT_EMBED_DIM,
        text_hidden_dim=TEXT_HIDDEN_DIM,
        text_output_dim=TEXT_OUTPUT_DIM,
        tabular_hidden_dims=TABULAR_HIDDEN_DIMS,
        tabular_output_dim=TABULAR_OUTPUT_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout=DROPOUT,
    )

    dummy_image = torch.randn(FAKE_BATCH_SIZE, 3, 224, 224)
    dummy_text = torch.randint(0, FAKE_VOCAB_SIZE, (FAKE_BATCH_SIZE, MAX_SYMPTOM_LEN))
    dummy_tabular = torch.randn(FAKE_BATCH_SIZE, FAKE_TABULAR_INPUT_DIM)

    logits = model(dummy_image, dummy_text, dummy_tabular)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Forward pass réussi. Sortie: {logits.shape} (attendu: ({FAKE_BATCH_SIZE}, {NUM_CLASSES}))")
    print(f"Paramètres totaux: {n_params:,} | Entraînables: {n_trainable:,}")


if __name__ == "__main__":
    main()
