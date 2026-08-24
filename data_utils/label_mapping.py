"""
label_mapping.py
==================
Les 3 sources de données (images, symptômes, tabulaire) utilisent probablement
des libellés de maladie différents. Ce module les unifie en classes canoniques
via une recherche de mots-clés (voir config.CANONICAL_CLASSES / LABEL_KEYWORDS).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CANONICAL_CLASSES, LABEL_KEYWORDS


def normalize_label(raw_label: str) -> str:
    """
    Convertit un libellé brut (ex: 'Lumpy_Skin_Disease', 'FMD', 'foot and mouth')
    en classe canonique ('healthy', 'lumpy_skin_disease', 'foot_and_mouth_disease').
    Retourne 'unknown' si aucun mot-clé ne correspond.
    """
    text = str(raw_label).strip().lower().replace("_", " ").replace("-", " ")

    for canonical_class, keywords in LABEL_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return canonical_class

    return "unknown"


CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


if __name__ == "__main__":
    # Petit test rapide
    tests = ["Healthy", "Lumpy_Skin_Disease", "Foot_and_Mouth_Disease",
             "lumpy", "fmd", "foot and mouth disease", "sain"]
    for t in tests:
        print(f"{t!r:35} -> {normalize_label(t)}")
