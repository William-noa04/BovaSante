"""
tabular_encoder.py
====================
MLP simple pour les données tabulaires déjà encodées/normalisées
(sortie de preprocess_tabular.py).
"""

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), output_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, output_dim)"""
        return self.mlp(x)


if __name__ == "__main__":
    model = TabularEncoder(input_dim=40, hidden_dims=(128, 64), output_dim=64)
    dummy = torch.randn(4, 40)
    out = model(dummy)
    print("Sortie TabularEncoder:", out.shape)  # attendu: (4, 64)
