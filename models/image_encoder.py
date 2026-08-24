"""
image_encoder.py
==================
CNN basé sur ResNet18 pré-entraîné (ImageNet), tête de classification
remplacée par une projection vers un vecteur d'embedding de dimension fixe.
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # On retire la dernière couche fully-connected (classification ImageNet)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 224, 224) -> (batch, output_dim)"""
        features = self.backbone(x)
        return self.projection(features)


if __name__ == "__main__":
    model = ImageEncoder(output_dim=256)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Sortie ImageEncoder:", out.shape)  # attendu: (4, 256)
