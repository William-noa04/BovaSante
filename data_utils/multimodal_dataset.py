"""
multimodal_dataset.py
=======================
Version finale, correspondant exactement au code qui a produit les résultats
d'entraînement validés (99.18% en fusion complète). Combine les 3 modalités
via un tirage aléatoire par classe canonique partagée (pas d'identifiant
commun entre les sources), avec des pools symptômes/tabulaire optionnels
pour permettre un split Train/Val/Test disjoint (évite la fuite de données).
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import RANDOM_SEED, MAX_SYMPTOM_LEN
from data_utils.label_mapping import normalize_label, CLASS_TO_IDX


class MultimodalCattleDataset(torch.utils.data.Dataset):
    def __init__(self, images_index_csv, symptoms_csv, tabular_csv, vocab,
                 symptoms_pool=None, tabular_pool=None, seed=RANDOM_SEED):
        """
        symptoms_pool / tabular_pool : sous-ensembles (train, val ou test) des
        DataFrames symptômes/tabulaire, déjà filtrés en amont (voir
        training/train.py pour la construction du split disjoint). Si None,
        charge et filtre l'intégralité des fichiers — à éviter pour un
        Dataset de validation/test si un autre split utilise déjà tout le pool.
        """
        self.rng = random.Random(seed)

        self.df_images = pd.read_csv(images_index_csv).copy()
        self.df_images["canonical_label"] = self.df_images["label"].apply(normalize_label)
        self.df_images = self.df_images[self.df_images["canonical_label"] != "unknown"].reset_index(drop=True)
        self.df_images["label_idx"] = self.df_images["canonical_label"].map(CLASS_TO_IDX)

        if symptoms_pool is not None:
            self.df_symptoms = symptoms_pool.reset_index(drop=True)
        else:
            self.df_symptoms = pd.read_csv(symptoms_csv)
            self.df_symptoms["canonical_label"] = self.df_symptoms["disease"].apply(normalize_label)
            self.df_symptoms = self.df_symptoms[self.df_symptoms["canonical_label"] != "unknown"].reset_index(drop=True)

        if tabular_pool is not None:
            self.df_tabular = tabular_pool.reset_index(drop=True)
        else:
            self.df_tabular = pd.read_csv(tabular_csv)
            self.df_tabular["canonical_label"] = self.df_tabular["Disease_Status"].apply(normalize_label)
            self.df_tabular = self.df_tabular[self.df_tabular["canonical_label"] != "unknown"].reset_index(drop=True)

        self.feature_cols = [c for c in self.df_tabular.columns
                              if c not in ("Disease_Status", "canonical_label", "target_encoded")]
        self.tabular_input_dim = len(self.feature_cols)

        self.symptoms_by_class = {
            c: self.df_symptoms[self.df_symptoms["canonical_label"] == c].index.tolist()
            for c in CLASS_TO_IDX
        }
        self.tabular_by_class = {
            c: self.df_tabular[self.df_tabular["canonical_label"] == c].index.tolist()
            for c in CLASS_TO_IDX
        }
        self.vocab = vocab

    def __len__(self):
        return len(self.df_images)

    def _encode_symptoms(self, text):
        return torch.tensor(self.vocab.encode(str(text), max_len=MAX_SYMPTOM_LEN), dtype=torch.long)

    def _sample_row(self, index_by_class, canonical_label):
        candidates = index_by_class.get(canonical_label, [])
        if not candidates:
            raise ValueError(f"Aucune ligne disponible pour la classe '{canonical_label}'")
        return self.rng.choice(candidates)

    def __getitem__(self, idx):
        row_img = self.df_images.iloc[idx]
        canonical_label = row_img["canonical_label"]

        with Image.open(row_img["image_path"]) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)
            image_array = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0

        symptom_idx = self._sample_row(self.symptoms_by_class, canonical_label)
        symptoms = self._encode_symptoms(self.df_symptoms.iloc[symptom_idx]["symptoms_clean"])

        tab_idx = self._sample_row(self.tabular_by_class, canonical_label)
        tabular = torch.tensor(
            self.df_tabular.iloc[tab_idx][self.feature_cols].astype(float).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        label = torch.tensor(int(row_img["label_idx"]), dtype=torch.long)

        return {
            "image": torch.from_numpy(image_array),
            "symptoms": symptoms,
            "tabular": tabular,
            "label": label,
        }
