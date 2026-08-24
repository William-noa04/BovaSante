"""
preprocess_images.py
=====================
- Parcourt les dossiers healthy / lumpy / foot-and-mouth
- Redimensionne chaque image en 224x224
- Sauvegarde les images redimensionnées dans data_processed/images_224/<classe>/
- Génère un CSV récapitulatif (image_path, label) utilisé ensuite par le Dataset PyTorch
"""

import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGES_DIR, IMAGES_OUT_DIR, IMG_SIZE, CLASS_MAP, OUTPUT_DIR

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def resize_and_save(src_path: str, dst_path: str) -> bool:
    """Ouvre une image, la convertit en RGB, la redimensionne et la sauvegarde."""
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            img.save(dst_path)
        return True
    except Exception as e:
        print(f"[WARN] Image illisible, ignorée: {src_path} ({e})")
        return False


def process_all_images():
    records = []

    for folder_name, class_label in CLASS_MAP.items():
        src_folder = os.path.join(IMAGES_DIR, folder_name)
        dst_folder = os.path.join(IMAGES_OUT_DIR, class_label)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.isdir(src_folder):
            print(f"[ERREUR] Dossier introuvable: {src_folder} — vérifie config.py")
            continue

        files = [f for f in os.listdir(src_folder) if f.lower().endswith(VALID_EXT)]
        print(f"-> {folder_name}: {len(files)} images trouvées")

        for fname in tqdm(files, desc=f"Traitement {folder_name}"):
            src_path = os.path.join(src_folder, fname)
            dst_path = os.path.join(dst_folder, fname)

            if resize_and_save(src_path, dst_path):
                records.append({
                    "image_path": dst_path,
                    "original_filename": fname,
                    "label": class_label,
                })

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUTPUT_DIR, "images_index.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ {len(df)} images traitées. Index sauvegardé: {out_csv}")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    process_all_images()
