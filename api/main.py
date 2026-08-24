"""
main.py
========
API FastAPI exposant deux endpoints de pré-diagnostic :

- POST /predict/multimodal    : image + symptômes + tabulaire -> healthy / lumpy / FMD
- POST /predict/tabular-only  : tabulaire seul -> toutes les classes de Disease_Status

Lancement local :
    uvicorn api.main:app --reload --port 8000

Documentation interactive une fois lancé :
    http://127.0.0.1:8000/docs
"""

import json
import sys
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MAX_SYMPTOM_LEN
from api.model_loader import registry
from api.inference_preprocessing import decode_image_bytes, preprocess_image, preprocess_symptoms_text, is_bovine_image
from api.schemas import (
    TabularInput, SimplifiedTabularInput, MultimodalPredictionResponse, TabularOnlyPredictionResponse,
    ClassProbability, HealthResponse,
)

app = FastAPI(
    title="API de pré-diagnostic bovin",
    description="Détection de la Dermatose Nodulaire Contagieuse Bovine et de la Fièvre Aphteuse "
                 "à partir d'images, de symptômes et de données physiologiques.",
    version="0.1.0",
)

# CORS ouvert pour faciliter les tests locaux depuis un frontend futur.
# À restreindre à des origines précises avant tout déploiement en production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models_on_startup():
    registry.load_all()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        multimodal_model_loaded=registry.multimodal_model is not None,
        tabular_only_model_loaded=registry.tabular_only_model is not None,
    )


@app.post("/predict/multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(
    image: UploadFile = File(..., description="Photographie de la lésion bovine"),
    symptoms: str = Form(..., description="Description textuelle des symptômes observés"),
    tabular_json: str = Form(
        ..., description="Données physiologiques/environnementales au format JSON (voir schéma TabularInput)"
    ),
):
    if registry.multimodal_model is None:
        raise HTTPException(status_code=503, detail="Le modèle multimodal n'est pas chargé (checkpoint manquant).")

    try:
        tabular_dict = json.loads(tabular_json)
        TabularInput(**tabular_dict)  # valide la présence/le type de tous les champs attendus
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Données tabulaires invalides: {e}")

    image_bytes = await image.read()
    try:
        pil_image = decode_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image invalide ou illisible: {e}")

    is_bovine, bovine_score = is_bovine_image(
        pil_image, registry.bovine_gate_model, registry.bovine_gate_transform, registry.device
    )
    if not is_bovine:
        raise HTTPException(
            status_code=422,
            detail="Cette image ne ressemble pas à une photo de bovin ou de lésion bovine. Réessayez avec une autre photo.",
        )

    image_tensor = preprocess_image(pil_image).to(registry.device)

    symptoms_tensor = preprocess_symptoms_text(symptoms, registry.vocab, MAX_SYMPTOM_LEN).to(registry.device)
    tabular_tensor = registry.tabular_preprocessor.transform(tabular_dict).to(registry.device)

    warning = None
    with torch.no_grad():
        logits = registry.multimodal_model(image_tensor, symptoms_tensor, tabular_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = int(torch.tensor(probs).argmax())
    predicted_class = registry.multimodal_classes[predicted_idx]
    confidence = probs[predicted_idx]

    if confidence < 0.5:
        warning = "Confiance faible : les trois classes sont proches, une consultation vétérinaire est recommandée."

    return MultimodalPredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=[
            ClassProbability(label=cls, probability=p)
            for cls, p in zip(registry.multimodal_classes, probs)
        ],
        warning=warning,
    )


@app.post("/predict/simplified", response_model=MultimodalPredictionResponse)
async def predict_simplified(
    image: UploadFile = File(..., description="Photographie de la lésion bovine"),
    symptoms: str = Form(..., description="Description textuelle des symptômes observés"),
    tabular_json: str = Form(
        ..., description="Sous-ensemble de données que l'éleveur peut fournir (voir schéma SimplifiedTabularInput)"
    ),
):
    """
    Diagnostic pensé pour un éleveur/particulier sans matériel de mesure ni
    connaissance des ~30 variables physiologiques du modèle : seuls l'âge,
    le pays, la région et les vaccins sont demandés, le reste est complété
    avec les valeurs par défaut (médianes/modes) calculées sur le dataset
    d'entraînement. Contrairement au chatbot, l'image et les symptômes fournis
    sont réellement utilisés par le modèle de fusion complet (pas la branche
    tabulaire seule, dont la fiabilité mesurée en isolation est d'environ 23%).
    """
    if registry.multimodal_model is None:
        raise HTTPException(status_code=503, detail="Le modèle multimodal n'est pas chargé (checkpoint manquant).")
    if registry.tabular_defaults is None:
        raise HTTPException(
            status_code=503,
            detail="Les valeurs par défaut sont introuvables. Lance compute_defaults.py puis redémarre le serveur.",
        )

    try:
        simplified_dict = json.loads(tabular_json)
        simplified = SimplifiedTabularInput(**simplified_dict)  # valide le sous-ensemble fourni par l'éleveur
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Données tabulaires invalides: {e}")

    merged_dict = {**registry.tabular_defaults, **simplified.model_dump()}
    try:
        TabularInput(**merged_dict)  # même garde-fou de complétude que /predict/multimodal
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Valeurs par défaut incomplètes côté serveur: {e}")

    image_bytes = await image.read()
    try:
        pil_image = decode_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image invalide ou illisible: {e}")

    is_bovine, bovine_score = is_bovine_image(
        pil_image, registry.bovine_gate_model, registry.bovine_gate_transform, registry.device
    )
    if not is_bovine:
        raise HTTPException(
            status_code=422,
            detail="Cette image ne ressemble pas à une photo de bovin ou de lésion bovine. Réessayez avec une autre photo.",
        )

    image_tensor = preprocess_image(pil_image).to(registry.device)

    symptoms_tensor = preprocess_symptoms_text(symptoms, registry.vocab, MAX_SYMPTOM_LEN).to(registry.device)
    tabular_tensor = registry.tabular_preprocessor.transform(merged_dict).to(registry.device)

    warning = None
    with torch.no_grad():
        logits = registry.multimodal_model(image_tensor, symptoms_tensor, tabular_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = int(torch.tensor(probs).argmax())
    predicted_class = registry.multimodal_classes[predicted_idx]
    confidence = probs[predicted_idx]

    if confidence < 0.5:
        warning = "Confiance faible : les trois classes sont proches, une consultation vétérinaire est recommandée."

    return MultimodalPredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=[
            ClassProbability(label=cls, probability=p)
            for cls, p in zip(registry.multimodal_classes, probs)
        ],
        warning=warning,
    )


@app.post("/predict/tabular-only", response_model=TabularOnlyPredictionResponse)
def predict_tabular_only(tabular: TabularInput):
    if registry.tabular_only_model is None:
        raise HTTPException(status_code=503, detail="Le modèle tabulaire seul n'est pas chargé (checkpoint manquant).")

    tabular_tensor = registry.tabular_preprocessor.transform(tabular.model_dump()).to(registry.device)

    with torch.no_grad():
        logits = registry.tabular_only_model(tabular_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = int(torch.tensor(probs).argmax())
    predicted_class = registry.tabular_only_classes[predicted_idx]
    confidence = probs[predicted_idx]

    return TabularOnlyPredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=[
            ClassProbability(label=cls, probability=p)
            for cls, p in zip(registry.tabular_only_classes, probs)
        ],
    )
