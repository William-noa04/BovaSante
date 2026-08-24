"""
config.py
=========
Centralise tous les chemins et paramètres du projet.
==> ADAPTE UNIQUEMENT CETTE SECTION à tes chemins réels sur ton PC.
"""

import os

# ---------------------------------------------------------------------------
# 1. CHEMINS BRUTS (à adapter)
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\user\Documents\VENV\ProjetSoutenance"

# Dossier contenant les 3 sous-dossiers d'images (healthy / lumpy / foot-and-mouth)
IMAGES_DIR = os.path.join(BASE_DIR, "Cows datasets")

# Fichier CSV symptômes : colonnes -> image, disease, symptoms, severity
SYMPTOMS_CSV = os.path.join(BASE_DIR, "symptoms_dataset.csv")  # <-- adapte le nom exact

# Fichier CSV tabulaire global
TABULAR_CSV = os.path.join(BASE_DIR, "global_cattle_disease_detection_dataset.csv")  # <-- adapte le nom exact

# ---------------------------------------------------------------------------
# 2. DOSSIERS DE SORTIE (générés automatiquement, ne pas toucher)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed")
IMAGES_OUT_DIR = os.path.join(OUTPUT_DIR, "images_224")
ENCODERS_DIR = os.path.join(OUTPUT_DIR, "encoders")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
os.makedirs(ENCODERS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. PARAMÈTRES
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)

# Noms de classes = noms des sous-dossiers réels chez toi
CLASS_MAP = {
    "healthy": "healthy",
    "lumpy": "lumpy",
    "foot-and-mouth": "foot-and-mouth",
}

# Colonnes du dataset tabulaire
TABULAR_TARGET_COL = "Disease_Status"

TABULAR_CATEGORICAL_COLS = [
    "Breed", "Region", "Country", "Climate_Zone", "Management_System",
    "Lactation_Stage", "Feed_Type", "Season",
    "FMD_Vaccine", "Brucellosis_Vaccine", "HS_Vaccine", "BQ_Vaccine",
    "Anthrax_Vaccine", "IBR_Vaccine", "BVD_Vaccine", "Rabies_Vaccine",
]

TABULAR_NUMERIC_COLS = [
    "Age_Months", "Weight_kg", "Parity", "Days_in_Milk", "Feed_Quantity_kg",
    "Water_Intake_L", "Walking_Distance_km", "Grazing_Duration_hrs",
    "Rumination_Time_hrs", "Resting_Hours", "Body_Temperature_C",
    "Heart_Rate_bpm", "Respiratory_Rate", "Ambient_Temperature_C",
    "Humidity_percent", "Housing_Score", "Milk_Yield_L",
    "Previous_Week_Avg_Yield", "Body_Condition_Score", "Milking_Interval_hrs",
]

# Colonnes à ignorer explicitement (identifiants, dates non exploitées telles quelles)
TABULAR_DROP_COLS = ["Cattle_ID", "Farm_ID", "Date"]

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 4. PARAMÈTRES DU MODÈLE MULTIMODAL (Étape 2)
# ---------------------------------------------------------------------------
# Classes canoniques communes aux 3 sources (unifiées par mots-clés, voir
# data_utils/label_mapping.py). Adapte les mots-clés si tes libellés diffèrent.
CANONICAL_CLASSES = ["healthy", "lumpy_skin_disease", "foot_and_mouth_disease"]
LABEL_KEYWORDS = {
    "healthy": ["healthy", "normal", "sain"],
    "lumpy_skin_disease": ["lumpy"],
    "foot_and_mouth_disease": ["foot", "fmd", "mouth"],
}
NUM_CLASSES = len(CANONICAL_CLASSES)

# Texte (symptômes)
MAX_SYMPTOM_LEN = 30      # nb max de mots par phrase de symptômes (padding/troncature)
TEXT_EMBED_DIM = 100
TEXT_HIDDEN_DIM = 128
TEXT_OUTPUT_DIM = 128

# Image
IMAGE_OUTPUT_DIM = 256

# Tabulaire
TABULAR_HIDDEN_DIMS = [128, 64]
TABULAR_OUTPUT_DIM = 64

# Fusion
FUSION_HIDDEN_DIM = 256
DROPOUT = 0.3

# Entraînement (utilisé à l'Étape 3)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

