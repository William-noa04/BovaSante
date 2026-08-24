"""
vocab.py
=========
Construit un vocabulaire simple (mot -> index) à partir de la colonne
'symptoms_clean' du CSV de symptômes, puis encode une phrase en séquence
d'entiers de longueur fixe (padding/troncature à MAX_SYMPTOM_LEN).

Tokens spéciaux :
  0 -> <PAD>
  1 -> <UNK>
"""

import os
import sys
import json
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MAX_SYMPTOM_LEN, ENCODERS_DIR

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1


class Vocab:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.itos = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(str(t).split())

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word
        return self

    def __len__(self):
        return len(self.stoi)

    def encode(self, text: str, max_len: int = MAX_SYMPTOM_LEN):
        """Convertit une phrase en liste d'entiers de longueur fixe (padding/troncature)."""
        tokens = str(text).split()[:max_len]
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens]
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.stoi = json.load(f)
        vocab.itos = {i: w for w, i in vocab.stoi.items()}
        return vocab


def build_and_save_vocab(symptoms_series, save_path=None):
    vocab = Vocab(min_freq=1).build(symptoms_series)
    save_path = save_path or os.path.join(ENCODERS_DIR, "symptoms_vocab.json")
    vocab.save(save_path)
    print(f"✅ Vocabulaire construit: {len(vocab)} mots -> {save_path}")
    return vocab


if __name__ == "__main__":
    import pandas as pd
    from config import OUTPUT_DIR

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "symptoms_clean.csv"))
    v = build_and_save_vocab(df["symptoms_clean"])
    print("Exemple d'encodage:", v.encode(df["symptoms_clean"].iloc[0]))
