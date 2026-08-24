"""
schemas.py
===========
Modèles Pydantic pour la validation des entrées/sorties de l'API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TabularInput(BaseModel):
    """
    Données tabulaires brutes attendues en entrée (avant tout encodage).
    Les noms de champs correspondent exactement aux colonnes du dataset
    original (voir config.py : TABULAR_CATEGORICAL_COLS / TABULAR_NUMERIC_COLS).
    """
    # Variables catégorielles
    Breed: str = Field(..., example="Holstein")
    Region: str = Field(..., example="Africa")
    Country: str = Field(..., example="Cameroon")
    Climate_Zone: str = Field(..., example="Tropical")
    Management_System: str = Field(..., example="Extensive")
    Lactation_Stage: str = Field(..., example="Mid")
    Feed_Type: str = Field(..., example="Pasture_Grass")
    Season: str = Field(..., example="Summer")
    FMD_Vaccine: int = Field(..., ge=0, le=1, example=1)
    Brucellosis_Vaccine: int = Field(..., ge=0, le=1, example=0)
    HS_Vaccine: int = Field(..., ge=0, le=1, example=0)
    BQ_Vaccine: int = Field(..., ge=0, le=1, example=0)
    Anthrax_Vaccine: int = Field(..., ge=0, le=1, example=0)
    IBR_Vaccine: int = Field(..., ge=0, le=1, example=0)
    BVD_Vaccine: int = Field(..., ge=0, le=1, example=0)
    Rabies_Vaccine: int = Field(..., ge=0, le=1, example=0)

    # Variables numériques
    Age_Months: float = Field(..., example=36)
    Weight_kg: float = Field(..., example=420)
    Parity: float = Field(..., example=2)
    Days_in_Milk: float = Field(..., example=120)
    Feed_Quantity_kg: float = Field(..., example=15)
    Water_Intake_L: float = Field(..., example=40)
    Walking_Distance_km: float = Field(..., example=3)
    Grazing_Duration_hrs: float = Field(..., example=6)
    Rumination_Time_hrs: float = Field(..., example=8)
    Resting_Hours: float = Field(..., example=10)
    Body_Temperature_C: float = Field(..., example=38.5)
    Heart_Rate_bpm: float = Field(..., example=70)
    Respiratory_Rate: float = Field(..., example=25)
    Ambient_Temperature_C: float = Field(..., example=28)
    Humidity_percent: float = Field(..., example=60)
    Housing_Score: float = Field(..., example=3)
    Milk_Yield_L: float = Field(..., example=12)
    Previous_Week_Avg_Yield: float = Field(..., example=12.5)
    Body_Condition_Score: float = Field(..., example=3)
    Milking_Interval_hrs: float = Field(..., example=12)


class SimplifiedTabularInput(BaseModel):
    """
    Sous-ensemble de TabularInput que peut raisonnablement fournir un éleveur
    sans expérience ni matériel de mesure. Le reste des champs est complété
    côté serveur avec les valeurs par défaut (tabular_defaults.json).
    """
    Age_Months: float = Field(..., example=12)
    Country: str = Field(..., example="Cameroon")
    Region: str = Field(..., example="Africa")
    FMD_Vaccine: int = Field(..., ge=0, le=1, example=1)
    Brucellosis_Vaccine: int = Field(..., ge=0, le=1, example=0)
    HS_Vaccine: int = Field(..., ge=0, le=1, example=0)
    BQ_Vaccine: int = Field(..., ge=0, le=1, example=0)
    Anthrax_Vaccine: int = Field(..., ge=0, le=1, example=0)
    IBR_Vaccine: int = Field(..., ge=0, le=1, example=0)
    BVD_Vaccine: int = Field(..., ge=0, le=1, example=0)
    Rabies_Vaccine: int = Field(..., ge=0, le=1, example=0)


class ClassProbability(BaseModel):
    label: str
    probability: float


class MultimodalPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: list[ClassProbability]
    warning: Optional[str] = None


class TabularOnlyPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: list[ClassProbability]


class HealthResponse(BaseModel):
    status: str
    multimodal_model_loaded: bool
    tabular_only_model_loaded: bool
