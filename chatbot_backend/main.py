"""
Backend FastAPI du chatbot vétérinaire bovin — BovaSanté.

Endpoint principal : POST /chat
  Body   : {"message": "...", "history": [...]}   (history = liste de messages précédents, optionnelle)
  Retour : {"reply": "...", "history": [...], "prediction": {...} | null}

Le chatbot répond aux questions générales d'élevage (comportement, alimentation,
premiers secours, mise bas, etc.) à partir de ses connaissances vétérinaires,
et invoque l'outil `predict_disease` lorsque l'éleveur décrit des signes cliniques
compatibles avec la DNCB ou la Fièvre Aphteuse, pour obtenir une estimation
chiffrée basée sur le modèle entraîné.

Moteur LLM : Google Gemini (API gratuite, palier gratuit sans carte bancaire,
voir https://ai.google.dev — modèle Flash, function calling supporté nativement).

Lancement :
    export GEMINI_API_KEY="AIza..."   (récupérée sur https://aistudio.google.com/apikey)
    export ENCODERS_DIR="C:\\Users\\user\\Documents\\VENV\\ProjetSoutenance\\data_processed\\encoders"
    export OUTPUT_DIR="C:\\Users\\user\\Documents\\VENV\\ProjetSoutenance\\data_processed"
    uvicorn main:app --reload --port 8000
"""
import os
import logging
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError

from inference import get_engine, InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bovasante-chat")

MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 1024

app = FastAPI(title="BovaSanté — Chatbot Vétérinaire")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production (ex: URL du frontend Streamlit)
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()  # lit GEMINI_API_KEY depuis l'environnement

_engine: Optional[InferenceEngine] = None


@app.on_event("startup")
def load_model_on_startup():
    global _engine
    try:
        _engine = get_engine()
        logger.info("Modèle et encoders chargés avec succès.")
    except FileNotFoundError as e:
        # On ne bloque pas le démarrage du serveur : le chat général fonctionne
        # quand même, seul l'outil predict_disease sera indisponible.
        logger.warning(f"Modèle non chargé — predict_disease sera indisponible.\n{e}")
        _engine = None


SYSTEM_PROMPT = """Tu es l'assistant vétérinaire de BovaSanté, une plateforme d'aide \
à la détection précoce de maladies bovines, développée pour des éleveurs camerounais.

DOMAINE STRICT : tu réponds UNIQUEMENT aux questions concernant la santé, le \
comportement, l'alimentation, la reproduction et les soins des bovins (vaches, \
taureaux, veaux). Pour toute question hors de ce domaine, décline poliment et \
recentre la conversation.

TES DEUX SPÉCIALITÉS DE DÉTECTION : la Dermatose Nodulaire Contagieuse Bovine \
(DNCB / Lumpy Skin Disease) et la Fièvre Aphteuse (FA / Foot-and-Mouth Disease). \
Tu peux aussi répondre à des questions générales d'élevage (mise bas, alimentation, \
premiers secours, autres pathologies courantes) à partir de tes connaissances \
vétérinaires générales.

QUAND UTILISER predict_disease : si l'éleveur décrit des signes cliniques \
observables sur son animal (fièvre, nodules sur la peau, boiterie, salivation \
excessive, lésions buccales ou aux sabots, baisse de production laitière, perte \
d'appétit...), pose UNE ou deux questions de clarification si nécessaire (température \
si connue, depuis combien de temps), puis appelle l'outil avec les informations \
disponibles. N'invente jamais de valeurs non mentionnées par l'éleveur — laisse-les \
vides, le système les complétera automatiquement.

APRÈS UNE PRÉDICTION : présente toujours le résultat comme une INDICATION, jamais \
un diagnostic définitif. Rappelle systématiquement de faire confirmer par un \
vétérinaire, en particulier si le résultat suggère une des deux maladies ciblées. \
Donne aussi des conseils de biosécurité immédiats et pratiques si pertinent \
(isolement de l'animal, désinfection, limitation des contacts).

TON : simple, direct, accessible à un éleveur non technique. Réponds en français. \
Pas de jargon médical non expliqué. Sois concis — pas de longs paragraphes inutiles."""


PREDICT_DISEASE_FUNCTION = types.FunctionDeclaration(
    name="predict_disease",
    description=(
        "Estime la probabilité que le bovin décrit soit sain, atteint de Dermatose "
        "Nodulaire Contagieuse Bovine (DNCB), ou de Fièvre Aphteuse (FA), à partir "
        "des paramètres physiologiques fournis par l'éleveur. Ne fournis que les "
        "champs que l'éleveur a mentionnés explicitement ou que tu peux estimer "
        "avec confiance à partir de sa description ; laisse les autres absents."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "Age_Months": {"type": "number", "description": "Âge de l'animal en mois"},
            "Weight_kg": {"type": "number", "description": "Poids en kg"},
            "Body_Temperature_C": {"type": "number", "description": "Température corporelle en °C (normale ~38.5°C)"},
            "Heart_Rate_bpm": {"type": "number", "description": "Fréquence cardiaque (battements/min)"},
            "Respiratory_Rate": {"type": "number", "description": "Fréquence respiratoire (mouvements/min)"},
            "Body_Condition_Score": {"type": "number", "description": "Note d'état corporel, échelle 1 (maigre) à 5 (gras)"},
            "Milk_Yield_L": {"type": "number", "description": "Production laitière actuelle en litres/jour"},
            "Previous_Week_Avg_Yield": {"type": "number", "description": "Moyenne de production laitière la semaine précédente (litres/jour)"},
            "FMD_Vaccine": {
                "type": "string",
                "description": "Statut vaccinal contre la fièvre aphteuse",
                "enum": ["yes", "no", "unknown"],
            },
        },
    },
)

PREDICT_DISEASE_TOOL = types.Tool(function_declarations=[PREDICT_DISEASE_FUNCTION])


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict[str, Any]]] = None


class ChatResponse(BaseModel):
    reply: str
    history: list[dict[str, Any]]
    prediction: Optional[dict[str, Any]] = None


def run_predict_disease_tool(tool_args: dict) -> dict:
    if _engine is None:
        return {
            "error": (
                "Le modèle de prédiction n'est pas disponible sur ce serveur "
                "(artefacts manquants ou non chargés). Réponds à partir de tes "
                "connaissances générales et recommande une consultation vétérinaire."
            )
        }
    return _engine.predict(tool_args)


# ----------------------------------------------------------------------
# Sérialisation de l'historique de conversation.
# On garde le format le plus simple possible côté client (JSON), et on
# reconstruit les objets types.Content propres au SDK Gemini à chaque appel.
# ----------------------------------------------------------------------
def dict_to_content(d: dict) -> types.Content:
    parts = []
    for p in d.get("parts", []):
        if "text" in p:
            parts.append(types.Part.from_text(text=p["text"]))
        elif "function_call" in p:
            parts.append(types.Part(function_call=types.FunctionCall(
                name=p["function_call"]["name"],
                args=p["function_call"]["args"],
            )))
        elif "function_response" in p:
            parts.append(types.Part.from_function_response(
                name=p["function_response"]["name"],
                response=p["function_response"]["response"],
            ))
    return types.Content(role=d["role"], parts=parts)


def content_to_dict(c: types.Content) -> dict:
    parts = []
    for p in c.parts or []:
        if p.text:
            parts.append({"text": p.text})
        elif p.function_call:
            parts.append({"function_call": {
                "name": p.function_call.name,
                "args": dict(p.function_call.args or {}),
            }})
        elif p.function_response:
            parts.append({"function_response": {
                "name": p.function_response.name,
                "response": dict(p.function_response.response or {}),
            }})
    return {"role": c.role, "parts": parts}


GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[PREDICT_DISEASE_TOOL],
    max_output_tokens=MAX_OUTPUT_TOKENS,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide.")

    contents: list[types.Content] = [dict_to_content(h) for h in (req.history or [])]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=req.message)]))

    prediction_result = None

    try:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=contents, config=GENERATE_CONFIG,
        )
    except APIError as e:
        logger.exception("Erreur API Gemini")
        raise HTTPException(status_code=502, detail=f"Erreur du service IA: {e}")

    # Boucle d'exécution d'outil (peut se répéter si le modèle enchaîne des appels)
    max_tool_turns = 5
    turns = 0
    while response.function_calls and turns < max_tool_turns:
        turns += 1
        # Le tour du modèle contenant le(s) appel(s) de fonction
        contents.append(response.candidates[0].content)

        function_response_parts = []
        for fc in response.function_calls:
            if fc.name == "predict_disease":
                result = run_predict_disease_tool(dict(fc.args or {}))
                if "error" not in result:
                    prediction_result = result
            else:
                result = {"error": f"Outil inconnu: {fc.name}"}
            function_response_parts.append(
                types.Part.from_function_response(name=fc.name, response=result)
            )

        contents.append(types.Content(role="tool", parts=function_response_parts))

        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=GENERATE_CONFIG,
            )
        except APIError as e:
            logger.exception("Erreur API Gemini (après function_call)")
            raise HTTPException(status_code=502, detail=f"Erreur du service IA: {e}")

    contents.append(response.candidates[0].content)

    reply_text = (response.text or "").strip()
    history_out = [content_to_dict(c) for c in contents]

    return ChatResponse(reply=reply_text, history=history_out, prediction=prediction_result)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _engine is not None}
