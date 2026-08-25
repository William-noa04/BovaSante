"""
config.py
=========

Centralise tous les chemins et paramètres du projet.

Les chemins sont construits automatiquement à partir de l'emplacement
du fichier config.py afin que le projet fonctionne aussi bien en local
sur Windows que sur Render/Linux.
"""

import os


# ---------------------------------------------------------------------------
# 1. CHEMINS DU PROJET
# ---------------------------------------------------------------------------

# Racine du projet.
# Sur Windows, cela donnera par exemple :
# C:\\Users\\user\\Documents\\VENV\\ProjetSoutenance
#
# Sur Render, cela donnera automatiquement quelque chose comme :
# /opt/render/project/src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Dossier contenant les datasets d'images
IMAGES_DIR = os.path.join(BASE_DIR, "Cows datasets")


# Fichier CSV symptômes
SYMPTOMS_CSV = os.path.join(
    BASE_DIR,
    "symptoms_dataset.csv"
)


# Fichier CSV tabulaire global
TABULAR_CSV = os.path.join(
    BASE_DIR,
    "global_cattle_disease_detection_dataset.csv"
)


# ---------------------------------------------------------------------------
# 2. DOSSIERS DE SORTIE / ARTEFACTS
# ---------------------------------------------------------------------------

# Dossier contenant les données prétraitées,
# les encodeurs et les modèles entraînés.
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data_processed"
)


# Images redimensionnées en 224x224
IMAGES_OUT_DIR = os.path.join(
    OUTPUT_DIR,
    "images_224"
)


# Encodeurs et autres artefacts nécessaires à l'inférence
ENCODERS_DIR = os.path.join(
    OUTPUT_DIR,
    "encoders"
)


# Création automatique des dossiers s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
os.makedirs(ENCODERS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. PARAMÈTRES
# ---------------------------------------------------------------------------

IMG_SIZE = (224, 224)


# Noms des classes correspondant aux dossiers d'images
CLASS_MAP = {
    "healthy": "healthy",
    "lumpy": "lumpy",
    "foot-and-mouth": "foot-and-mouth",
}


# ---------------------------------------------------------------------------
# 4. PARAMÈTRES DU DATASET TABULAIRE
# ---------------------------------------------------------------------------

TABULAR_TARGET_COL = "Disease_Status"


TABULAR_CATEGORICAL_COLS = [
    "Breed",
    "Region",
    "Country",
    "Climate_Zone",
    "Management_System",
    "Lactation_Stage",
    "Feed_Type",
    "Season",
    "FMD_Vaccine",
    "Brucellosis_Vaccine",
    "HS_Vaccine",
    "BQ_Vaccine",
    "Anthrax_Vaccine",
    "IBR_Vaccine",
    "BVD_Vaccine",
    "Rabies_Vaccine",
]


TABULAR_NUMERIC_COLS = [
    "Age_Months",
    "Weight_kg",
    "Parity",
    "Days_in_Milk",
    "Feed_Quantity_kg",
    "Water_Intake_L",
    "Walking_Distance_km",
    "Grazing_Duration_hrs",
    "Rumination_Time_hrs",
    "Resting_Hours",
    "Body_Temperature_C",
    "Heart_Rate_bpm",
    "Respiratory_Rate",
    "Ambient_Temperature_C",
    "Humidity_percent",
    "Housing_Score",
    "Milk_Yield_L",
    "Previous_Week_Avg_Yield",
    "Body_Condition_Score",
    "Milking_Interval_hrs",
]


# Colonnes ignorées explicitement
TABULAR_DROP_COLS = [
    "Cattle_ID",
    "Farm_ID",
    "Date",
]


RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 5. PARAMÈTRES DU MODÈLE MULTIMODAL
# ---------------------------------------------------------------------------

# Classes canoniques communes aux différentes sources
CANONICAL_CLASSES = [
    "healthy",
    "lumpy_skin_disease",
    "foot_and_mouth_disease",
]


LABEL_KEYWORDS = {
    "healthy": [
        "healthy",
        "normal",
        "sain",
    ],

    "lumpy_skin_disease": [
        "lumpy",
    ],

    "foot_and_mouth_disease": [
        "foot",
        "fmd",
        "mouth",
    ],
}


NUM_CLASSES = len(CANONICAL_CLASSES)


# ---------------------------------------------------------------------------
# 6. PARAMÈTRES TEXTE / SYMPTÔMES
# ---------------------------------------------------------------------------

# Nombre maximum de mots dans une phrase de symptômes
MAX_SYMPTOM_LEN = 30

TEXT_EMBED_DIM = 100
TEXT_HIDDEN_DIM = 128
TEXT_OUTPUT_DIM = 128


# ---------------------------------------------------------------------------
# 7. PARAMÈTRES IMAGE
# ---------------------------------------------------------------------------

IMAGE_OUTPUT_DIM = 256


# ---------------------------------------------------------------------------
# 8. PARAMÈTRES TABULAIRES
# ---------------------------------------------------------------------------

TABULAR_HIDDEN_DIMS = [
    128,
    64,
]

TABULAR_OUTPUT_DIM = 64


# ---------------------------------------------------------------------------
# 9. PARAMÈTRES DE FUSION MULTIMODALE
# -----------------------------------------------------------------------

FUSION_HIDDEN_DIM = 256

DROPOUT = 0.3


# ---------------------------------------------------------------------------
# 10. PARAMÈTRES D'ENTRAÎNEMENT
# ---------------------------------------------------------------------------

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

NUM_EPOCHS = 30