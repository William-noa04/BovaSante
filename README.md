# BovaSanté — Chatbot Vétérinaire (backend)

## 1. Installation

```bash
cd chemin\vers\ce\dossier
pip install -r requirements.txt
```

## 2. Préparer les artefacts (une seule fois)

Ce chatbot réutilise le modèle et les encoders déjà générés par ton notebook de
prétraitement/entraînement (`data_processed/`). Il te manque juste un fichier :
les valeurs par défaut pour les champs que l'éleveur ne fournira pas en
conversation (médianes/modes du dataset).

```bash
python compute_defaults.py
```

Vérifie que `data_processed/encoders/` contient bien, à la fin :
- `tabular_schema.json`
- `tabular_scaler.pkl`
- `tabular_onehot_encoder.pkl`
- `tabular_label_encoders.pkl`
- `tabular_target_encoder.pkl`
- `symptoms_vocab.json`
- `tabular_defaults.json` (nouveau, généré à l'instant)

Et que `data_processed/best_model.pt` existe (généré à l'Étape 3 du notebook
d'entraînement).

## 3. Clé API Gemini (gratuite)

1. Va sur **https://aistudio.google.com/apikey**
2. Connecte-toi avec un compte Google
3. Clique sur **Create API key** (aucune carte bancaire requise)
4. Copie la clé (commence par `AIza...`)

Palier gratuit (modèle `gemini-2.5-flash`, utilisé par défaut ici) : environ
15 requêtes/minute et 1500/jour — largement suffisant pour développer et
présenter ce projet.

```powershell
# PowerShell (Windows)
$env:GEMINI_API_KEY="AIza..."   # ta propre clé, récupérée sur https://aistudio.google.com/apikey
$env:ENCODERS_DIR="chemin\vers\ce\dossier\data_processed\encoders"
$env:OUTPUT_DIR="chemin\vers\ce\dossier\data_processed"
```

## 4. Lancer le serveur

```bash
uvicorn main:app --reload --port 8000
```

Vérifie que tout est chargé : http://localhost:8000/health
Doit répondre : `{"status": "ok", "model_loaded": true}`

## 5. Tester

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Mon animal ne mange plus depuis trois jours, il a de la fievre et des boutons sur la peau\"}"
```

Ou une question générale sans prédiction :

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Comment faire vêler une vache en difficulté ?\"}"
```

Pour poursuivre une conversation, renvoie le champ `history` reçu dans la
réponse précédente comme `history` de la requête suivante.

## 6. Notes importantes

- La prédiction de `predict_disease` utilise uniquement la **branche tabulaire**
  du modèle de fusion (image et texte mis à zéro), car le chatbot ne dispose ni
  d'image ni de description symptomatique structurée. La fiabilité mesurée en
  ablation pour cette branche seule est d'environ **23 %** — c'est documenté
  dans le prompt système, et le chatbot est instruit de toujours présenter le
  résultat comme une indication à confirmer par un vétérinaire.
- Si `data_processed/encoders/tabular_defaults.json` ou `best_model.pt` sont
  absents au démarrage, le serveur démarre quand même (chat général
  fonctionnel) mais `predict_disease` renverra une erreur explicite.
- CORS est ouvert à `*` pour le développement — à restreindre à l'URL exacte
  du frontend Streamlit avant toute mise en production.
