"""
text_encoder.py
=================
Encodeur de texte pour les symptômes : Embedding entraînable + BiLSTM,
suivi d'une projection vers un vecteur d'embedding de dimension fixe.
Le pooling utilise le dernier état caché des deux directions.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128,
                 output_dim: int = 128, num_layers: int = 1, padding_idx: int = 0,
                 dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) -> (batch, output_dim)"""
        embedded = self.embedding(token_ids)              # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)                 # h_n: (2*num_layers, batch, hidden_dim)

        # Concatène les derniers états cachés forward + backward de la dernière couche
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)

        return self.projection(h_concat)


if __name__ == "__main__":
    model = TextEncoder(vocab_size=500, embed_dim=100, hidden_dim=128, output_dim=128)
    dummy = torch.randint(0, 500, (4, 30))  # (batch=4, seq_len=30)
    out = model(dummy)
    print("Sortie TextEncoder:", out.shape)  # attendu: (4, 128)
