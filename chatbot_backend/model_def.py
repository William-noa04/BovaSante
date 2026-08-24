"""
Définitions des classes du modèle multimodal de détection des maladies bovines.

IMPORTANT : ces classes doivent rester STRICTEMENT identiques à celles utilisées
lors de l'entraînement (notebook Étape 2/3), sinon torch.load_state_dict() échouera
ou chargera des poids dans la mauvaise couche silencieusement.
"""
import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim), nn.ReLU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=128,
                 num_layers=1, padding_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, (h_n, _) = self.lstm(embedded)
        h_forward, h_backward = h_n[-2, :, :], h_n[-1, :, :]
        h_concat = torch.cat([h_forward, h_backward], dim=1)
        return self.projection(h_concat)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=64, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultimodalFusionModel(nn.Module):
    def __init__(self, vocab_size, tabular_input_dim, num_classes,
                 image_output_dim=256, text_embed_dim=100, text_hidden_dim=128,
                 text_output_dim=128, tabular_hidden_dims=(128, 64),
                 tabular_output_dim=64, fusion_hidden_dim=256, dropout=0.3):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=image_output_dim)
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, text_hidden_dim, text_output_dim)
        self.tabular_encoder = TabularEncoder(tabular_input_dim, tabular_hidden_dims, tabular_output_dim)

        fused_dim = image_output_dim + text_output_dim + tabular_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def forward(self, image, symptoms, tabular):
        img_emb = self.image_encoder(image)
        text_emb = self.text_encoder(symptoms)
        tab_emb = self.tabular_encoder(tabular)
        fused = torch.cat([img_emb, text_emb, tab_emb], dim=1)
        return self.classifier(fused)
