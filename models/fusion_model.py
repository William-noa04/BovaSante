"""
fusion_model.py
=================
Modèle multimodal complet :
  image  -> ImageEncoder   -> vecteur (image_dim)
  texte  -> TextEncoder    -> vecteur (text_dim)
  tabul. -> TabularEncoder -> vecteur (tabular_dim)

Fusion = concaténation des 3 vecteurs, suivie d'un MLP de classification.
"""

import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.tabular_encoder import TabularEncoder


class MultimodalFusionModel(nn.Module):
    def __init__(self, vocab_size: int, tabular_input_dim: int, num_classes: int,
                 image_output_dim: int = 256, text_embed_dim: int = 100,
                 text_hidden_dim: int = 128, text_output_dim: int = 128,
                 tabular_hidden_dims=(128, 64), tabular_output_dim: int = 64,
                 fusion_hidden_dim: int = 256, dropout: float = 0.3,
                 freeze_image_backbone: bool = False):
        super().__init__()

        self.image_encoder = ImageEncoder(
            output_dim=image_output_dim, pretrained=True,
            freeze_backbone=freeze_image_backbone,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim, output_dim=text_output_dim,
        )
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim, hidden_dims=tabular_hidden_dims,
            output_dim=tabular_output_dim,
        )

        fused_dim = image_output_dim + text_output_dim + tabular_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def forward(self, image: torch.Tensor, symptoms: torch.Tensor,
                tabular: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_encoder(image)          # (batch, image_output_dim)
        text_emb = self.text_encoder(symptoms)        # (batch, text_output_dim)
        tab_emb = self.tabular_encoder(tabular)        # (batch, tabular_output_dim)

        fused = torch.cat([img_emb, text_emb, tab_emb], dim=1)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    model = MultimodalFusionModel(
        vocab_size=500, tabular_input_dim=40, num_classes=3,
    )
    dummy_image = torch.randn(4, 3, 224, 224)
    dummy_text = torch.randint(0, 500, (4, 30))
    dummy_tabular = torch.randn(4, 40)

    logits = model(dummy_image, dummy_text, dummy_tabular)
    print("Sortie MultimodalFusionModel:", logits.shape)  # attendu: (4, 3)
