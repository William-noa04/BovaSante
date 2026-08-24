"""
tabular_only_model.py
========================
Modèle utilisant UNIQUEMENT les données tabulaires (pas d'image, pas de texte).
Permet de prédire l'ensemble des maladies présentes dans Disease_Status,
y compris celles qui n'ont pas d'images ni de symptômes associés.
"""

import torch
import torch.nn as nn

from models.tabular_encoder import TabularEncoder


class TabularOnlyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims=(128, 64), embedding_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = TabularEncoder(
            input_dim=input_dim, hidden_dims=hidden_dims,
            output_dim=embedding_dim, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return self.classifier(embedding)


if __name__ == "__main__":
    model = TabularOnlyClassifier(input_dim=40, num_classes=7)
    dummy = torch.randn(8, 40)
    out = model(dummy)
    print("Sortie TabularOnlyClassifier:", out.shape)  # attendu: (8, 7)
