"""
streamlit_app.py
==================
Application de pré-diagnostic bovin — interface pensée pour des éleveurs
(vocabulaire simple, verdict visuel clair, formulaire organisé par sections).

Lancement :
    streamlit run streamlit_app.py
"""

import os
import sys

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import MAX_SYMPTOM_LEN
from api.model_loader import ModelRegistry
from api.inference_preprocessing import preprocess_image_bytes, preprocess_symptoms_text

# ---------------------------------------------------------------------------
# Personnalisation rapide : change juste ces deux lignes pour rebrander l'appli
# ---------------------------------------------------------------------------
APP_NAME = "BovaSanté"
APP_TAGLINE = "Le compagnon numérique de votre troupeau"

# Messages de recommandation par maladie (à adapter/enrichir librement)
RECOMMENDATIONS = {
    "healthy": {
        "icon": "✅",
        "title": "Aucun signe de maladie détecté",
        "message": "L'animal ne présente pas de signes des maladies surveillées par l'application. "
                    "Continuez une surveillance régulière du troupeau.",
        "level": "success",
    },
    "lumpy_skin_disease": {
        "icon": "⚠️",
        "title": "Signes compatibles avec la Dermatose Nodulaire Contagieuse Bovine",
        "message": "Isolez l'animal du reste du troupeau dès que possible et contactez un agent "
                    "vétérinaire. Cette maladie est très contagieuse.",
        "level": "warning",
    },
    "foot_and_mouth_disease": {
        "icon": "🚨",
        "title": "Signes compatibles avec la Fièvre Aphteuse",
        "message": "Maladie à déclaration obligatoire. Isolez immédiatement le troupeau, limitez "
                    "les déplacements d'animaux et de personnes, et alertez sans délai les services "
                    "vétérinaires les plus proches.",
        "level": "error",
    },
}
DEFAULT_RECOMMENDATION = {
    "icon": "🔎",
    "title": "Résultat à confirmer",
    "message": "Ce résultat est une orientation. Contactez un agent vétérinaire pour confirmer le diagnostic.",
    "level": "info",
}

st.set_page_config(page_title=APP_NAME, page_icon="🐄", layout="wide")

# ---------------------------------------------------------------------------
# Thème visuel — palette marron / terre, pensée pour un univers agricole
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --brown-darkest: #3E2723;
        --brown-dark: #5D4037;
        --brown-medium: #8D6E63;
        --brown-light: #D7CCC8;
        --brown-lightest: #EFEBE9;
        --accent: #C08B3E;
        --accent-dark: #9C6B26;
        --cream: #FBF7F2;
    }

    .stApp {
        background-color: var(--cream);
    }

    .app-header {
        background: linear-gradient(135deg, var(--brown-darkest), var(--brown-dark));
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(62, 39, 35, 0.25);
    }
    .app-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2rem;
    }
    .app-header p {
        color: var(--brown-light);
        margin: 0.3rem 0 0 0;
        font-size: 1.05rem;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--brown-light);
        border-left: 5px solid var(--accent);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    h2, h3 {
        color: var(--brown-darkest) !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        background-color: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: var(--accent-dark);
        color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--brown-lightest);
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        color: var(--brown-dark);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--brown-medium) !important;
        color: #FFFFFF !important;
    }

    .verdict-card {
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        border-left: 8px solid var(--accent);
    }
    .verdict-card.success { background-color: #E8F5E9; border-left-color: #4CAF50; }
    .verdict-card.warning { background-color: #FFF3E0; border-left-color: #FF9800; }
    .verdict-card.error   { background-color: #FFEBEE; border-left-color: #F44336; }
    .verdict-card.info    { background-color: var(--brown-lightest); border-left-color: var(--brown-medium); }
    .verdict-card h3 { margin-top: 0; }

    .app-footer {
        text-align: center;
        color: var(--brown-medium);
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--brown-light);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.load_all()
    return registry


def render_tabular_form(registry: ModelRegistry, key_prefix: str) -> dict:
    """
    Formulaire tabulaire organisé en sections thématiques repliables, avec les
    options générées dynamiquement à partir des encodeurs réellement appris
    par le modèle (aucune valeur inventée qui tomberait en "inconnue").
    """
    schema = registry.tabular_preprocessor.schema
    low_card_cols = schema["low_card_cols"]
    high_card_cols = schema["high_card_cols"]
    numeric_cols = schema["numeric_cols"]

    ohe = registry.tabular_preprocessor.onehot_encoder
    label_encoders = registry.tabular_preprocessor.label_encoders

    vaccine_cols = [c for c in low_card_cols if "Vaccine" in c]
    env_low_cols = [c for c in low_card_cols if c not in vaccine_cols]

    raw_data = {}

    with st.expander("🐮  Informations sur l'animal", expanded=True):
        cols_ui = st.columns(3)
        for i, col in enumerate(high_card_cols):
            le = label_encoders.get(col)
            options = list(le.classes_) if le is not None else [""]
            with cols_ui[i % 3]:
                label = col.replace("_", " ")
                raw_data[col] = st.selectbox(label, options, key=f"{key_prefix}_{col}")

        numeric_animal = [c for c in numeric_cols if c in (
            "Age_Months", "Weight_kg", "Parity", "Days_in_Milk",
            "Body_Condition_Score", "Milk_Yield_L", "Previous_Week_Avg_Yield",
        )]
        cols_ui_num = st.columns(3)
        for i, col in enumerate(numeric_animal):
            with cols_ui_num[i % 3]:
                raw_data[col] = st.number_input(col.replace("_", " "), value=0.0, key=f"{key_prefix}_{col}")

    with st.expander("🌡️  Mesures physiologiques"):
        physio_cols = [c for c in numeric_cols if c in (
            "Body_Temperature_C", "Heart_Rate_bpm", "Respiratory_Rate",
        )]
        cols_ui = st.columns(3)
        for i, col in enumerate(physio_cols):
            with cols_ui[i % 3]:
                raw_data[col] = st.number_input(col.replace("_", " "), value=0.0, key=f"{key_prefix}_{col}")

    with st.expander("🌍  Environnement et élevage"):
        cols_ui = st.columns(3)
        for i, col in enumerate(env_low_cols):
            options = list(ohe.categories_[low_card_cols.index(col)]) if ohe is not None else ["0", "1"]
            with cols_ui[i % 3]:
                raw_data[col] = st.selectbox(col.replace("_", " "), options, key=f"{key_prefix}_{col}")

        env_numeric = [c for c in numeric_cols if c in (
            "Water_Intake_L", "Walking_Distance_km", "Grazing_Duration_hrs",
            "Rumination_Time_hrs", "Resting_Hours", "Feed_Quantity_kg",
            "Ambient_Temperature_C", "Humidity_percent", "Housing_Score",
            "Milking_Interval_hrs",
        )]
        cols_ui2 = st.columns(3)
        for i, col in enumerate(env_numeric):
            with cols_ui2[i % 3]:
                raw_data[col] = st.number_input(col.replace("_", " "), value=0.0, key=f"{key_prefix}_{col}")

    with st.expander("💉  Statut vaccinal"):
        st.caption("Sélectionnez « 1 » si l'animal est vacciné, « 0 » sinon.")
        cols_ui = st.columns(4)
        for i, col in enumerate(vaccine_cols):
            options = list(ohe.categories_[low_card_cols.index(col)]) if ohe is not None else ["0", "1"]
            with cols_ui[i % 4]:
                label = col.replace("_Vaccine", "").replace("_", " ")
                raw_data[col] = st.selectbox(label, options, key=f"{key_prefix}_{col}")

    return raw_data


def show_verdict(predicted_class: str, confidence: float):
    info = RECOMMENDATIONS.get(predicted_class, DEFAULT_RECOMMENDATION)
    st.markdown(f"""
    <div class="verdict-card {info['level']}">
        <h3>{info['icon']} {info['title']}</h3>
        <p>{info['message']}</p>
        <p><strong>Niveau de confiance du modèle : {confidence:.0%}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 0.5:
        st.info("La confiance du modèle est faible sur ce cas : les signes observés sont ambigus. "
                 "Une consultation vétérinaire est particulièrement recommandée.")


def show_details(classes: list, probs: list):
    with st.expander("📊 Voir le détail technique du résultat"):
        df = pd.DataFrame({"Classe": classes, "Probabilité": probs}).sort_values(
            "Probabilité", ascending=False
        )
        st.bar_chart(df.set_index("Classe"))
        st.dataframe(df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <h1>🐄 {APP_NAME}</h1>
    <p>{APP_TAGLINE}</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Chargement des modèles..."):
    registry = get_registry()

col_status1, col_status2 = st.columns(2)
col_status1.metric(
    "Diagnostic avec photo",
    "Prêt ✅" if registry.multimodal_model else "Indisponible ❌",
)
col_status2.metric(
    "Diagnostic sans photo",
    "Prêt ✅" if registry.tabular_only_model else "Indisponible ❌",
)

st.write("")

tab1, tab2 = st.tabs(["📷  Diagnostic avec photo", "📋  Diagnostic sans photo"])

# --- Onglet 1 : diagnostic complet (image + symptômes + tabulaire) ---
with tab1:
    st.subheader("Diagnostic complet")
    st.caption(
        "Le plus précis : prenez une photo de la lésion, décrivez les symptômes, "
        "et renseignez les informations sur l'animal."
    )

    if registry.multimodal_model is None:
        st.error("Ce module n'est pas encore disponible sur cet appareil.")
    else:
        with st.form("multimodal_form"):
            image_file = st.file_uploader(
                "📷 Photo de la lésion (peau, bouche ou pieds)",
                type=["jpg", "jpeg", "png", "bmp"],
            )
            symptoms_text = st.text_area(
                "📝 Décrivez ce que vous observez",
                placeholder="ex : perte d'appétit, boiterie, écoulement des yeux, fièvre...",
            )
            st.markdown("---")
            tabular_data = render_tabular_form(registry, key_prefix="mm")
            submitted = st.form_submit_button("🔍 Lancer le diagnostic")

        if submitted:
            if image_file is None:
                st.error("Merci d'ajouter une photo de l'animal.")
            elif not symptoms_text.strip():
                st.error("Merci de décrire les symptômes observés.")
            else:
                with st.spinner("Analyse en cours..."):
                    image_tensor = preprocess_image_bytes(image_file.read()).to(registry.device)
                    symptoms_tensor = preprocess_symptoms_text(
                        symptoms_text, registry.vocab, MAX_SYMPTOM_LEN
                    ).to(registry.device)
                    tabular_tensor = registry.tabular_preprocessor.transform(tabular_data).to(registry.device)

                    with torch.no_grad():
                        logits = registry.multimodal_model(image_tensor, symptoms_tensor, tabular_tensor)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

                predicted_idx = int(torch.tensor(probs).argmax())
                predicted_class = registry.multimodal_classes[predicted_idx]
                confidence = probs[predicted_idx]

                show_verdict(predicted_class, confidence)
                show_details(registry.multimodal_classes, probs)

# --- Onglet 2 : diagnostic rapide (tabulaire seul, toutes les maladies) ---
with tab2:
    st.subheader("Diagnostic rapide")
    st.caption(
        "Sans photo ni description : à partir des informations sur l'animal uniquement. "
        "Ce module couvre un plus grand nombre de maladies, mais est moins précis que le "
        "diagnostic complet."
    )

    if registry.tabular_only_model is None:
        st.error("Ce module n'est pas encore disponible sur cet appareil.")
    else:
        with st.form("tabular_only_form"):
            tabular_data_2 = render_tabular_form(registry, key_prefix="tab")
            submitted_2 = st.form_submit_button("🔍 Lancer le diagnostic")

        if submitted_2:
            with st.spinner("Analyse en cours..."):
                tabular_tensor = registry.tabular_preprocessor.transform(tabular_data_2).to(registry.device)
                with torch.no_grad():
                    logits = registry.tabular_only_model(tabular_tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

            predicted_idx = int(torch.tensor(probs).argmax())
            predicted_class = registry.tabular_only_classes[predicted_idx]
            confidence = probs[predicted_idx]

            show_verdict(predicted_class, confidence)
            show_details(registry.tabular_only_classes, probs)

st.markdown(f"""
<div class="app-footer">
    {APP_NAME} fournit une orientation de pré-diagnostic, pas un diagnostic médical certifié.<br>
    En cas de doute, consultez toujours un agent ou un service vétérinaire.
</div>
""", unsafe_allow_html=True)