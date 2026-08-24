"""
train_tabular_only.py
========================
Entraîne le modèle tabulaire seul sur TOUTES les classes de Disease_Status
présentes dans le dataset tabulaire (y compris celles sans image/symptômes).
"""

import os
import sys
import copy

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OUTPUT_DIR, ENCODERS_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from models.tabular_only_model import TabularOnlyClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_tabular_only_model.pt")
EARLY_STOPPING_PATIENCE = 5


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "tabular_clean.csv"))
    target_encoder = joblib.load(os.path.join(ENCODERS_DIR, "tabular_target_encoder.pkl"))

    feature_cols = [c for c in df.columns if c not in ("Disease_Status", "target_encoded")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target_encoded"].values.astype(np.int64)

    return X, y, feature_cols, target_encoder


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def train():
    X, y, feature_cols, target_encoder = load_data()
    num_classes = len(target_encoder.classes_)
    print(f"Classes détectées ({num_classes}):", list(target_encoder.classes_))

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    full_dataset = TensorDataset(X_tensor, y_tensor)

    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=RANDOM_SEED
    )

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = TabularOnlyClassifier(input_dim=X.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Époque {epoch:02d}/{NUM_EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"⏹ Early stopping à l'époque {epoch}")
                break

    model.load_state_dict(best_state)

    # Évaluation finale
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy Test (tabulaire seul, {num_classes} classes): {acc:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=target_encoder.classes_))
    print("Matrice de confusion:")
    print(confusion_matrix(all_labels, all_preds))

    return model, target_encoder


if __name__ == "__main__":
    train()
