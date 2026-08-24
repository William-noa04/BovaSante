"""
train.py
=========
Étape 3 : Entraînement du modèle multimodal.

- Split Train / Validation / Test (stratifié par classe)
- Fonction de perte : CrossEntropyLoss
- Optimiseur : Adam
- Boucle d'entraînement avec early stopping + sauvegarde du meilleur modèle
- Évaluation finale sur le test set (accuracy, classification report, matrice de confusion)
"""

import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    OUTPUT_DIR, ENCODERS_DIR, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, RANDOM_SEED, CANONICAL_CLASSES,
)
from data_utils.vocab import Vocab, build_and_save_vocab
from data_utils.multimodal_dataset import MultimodalCattleDataset
from models.fusion_model import MultimodalFusionModel

import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
EARLY_STOPPING_PATIENCE = 5


def build_dataset():
    """Construit le vocabulaire (si besoin) et le Dataset multimodal complet."""
    symptoms_csv = os.path.join(OUTPUT_DIR, "symptoms_clean.csv")
    tabular_csv = os.path.join(OUTPUT_DIR, "tabular_clean.csv")
    images_index_csv = os.path.join(OUTPUT_DIR, "images_index.csv")

    vocab_path = os.path.join(ENCODERS_DIR, "symptoms_vocab.json")
    if os.path.exists(vocab_path):
        vocab = Vocab.load(vocab_path)
    else:
        df_symptoms = pd.read_csv(symptoms_csv)
        vocab = build_and_save_vocab(df_symptoms["symptoms_clean"], vocab_path)

    dataset = MultimodalCattleDataset(
        images_index_csv=images_index_csv,
        symptoms_csv=symptoms_csv,
        tabular_csv=tabular_csv,
        vocab=vocab,
    )
    return dataset, vocab


def split_dataset(dataset):
    """Split stratifié Train (70%) / Validation (15%) / Test (15%) par classe."""
    labels = dataset.df_images["canonical_label"].values
    indices = np.arange(len(dataset))

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )

    print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def run_epoch(model, loader, criterion, optimizer=None):
    """Exécute une époque. Si optimizer est fourni -> mode entraînement, sinon évaluation."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            images = batch["image"].to(DEVICE)
            symptoms = batch["symptoms"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(images, symptoms, tabular)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def train():
    dataset, vocab = build_dataset()
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalFusionModel(
        vocab_size=len(vocab),
        tabular_input_dim=dataset.tabular_input_dim,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Époque {epoch:02d}/{NUM_EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_model_state, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"⏹ Early stopping à l'époque {epoch} (pas d'amélioration depuis {EARLY_STOPPING_PATIENCE} époques)")
                break

    # Recharge le meilleur modèle avant l'évaluation finale
    model.load_state_dict(best_model_state)
    evaluate_on_test(model, test_loader)
    return model, history


def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            symptoms = batch["symptoms"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images, symptoms, tabular)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy sur le Test set: {acc:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=CANONICAL_CLASSES))
    print("Matrice de confusion:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    train()
