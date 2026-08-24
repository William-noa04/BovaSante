"""
run_preprocessing.py
======================
Lance l'ensemble de l'Étape 1 : prétraitement des 3 sources de données.

Usage:
    python run_preprocessing.py
"""

from preprocessing.preprocess_images import process_all_images
from preprocessing.preprocess_symptoms import process_symptoms
from preprocessing.preprocess_tabular import process_tabular

if __name__ == "__main__":
    print("=" * 60)
    print("1/3 - Prétraitement des IMAGES")
    print("=" * 60)
    process_all_images()

    print("\n" + "=" * 60)
    print("2/3 - Prétraitement des SYMPTÔMES")
    print("=" * 60)
    process_symptoms()

    print("\n" + "=" * 60)
    print("3/3 - Prétraitement du TABULAIRE")
    print("=" * 60)
    process_tabular()

    print("\n✅ Étape 1 terminée. Résultats dans data_processed/")
