"""
API de scoring crédit "Prêt à Dépenser" (FastAPI).

Lancement en local :
    uvicorn src.api:app --reload
Documentation interactive (Swagger) : http://127.0.0.1:8000/docs
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.predict import THRESHOLD, get_features, load_model, predict

# Chargement du modèle UNE SEULE FOIS, au démarrage de l'API, puis réutilisé par toutes les requêtes
# (le recharger à chaque requête ralentirait fortement l'API)
model = load_model()
FEATURES = get_features(model)

# Client d'exemple, affiché pré-rempli dans Swagger pour tester l'API en un clic
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "client_exemple.json"
with open(EXAMPLE_PATH) as f:
    EXAMPLE_CLIENT = json.load(f)


# ---------------------------------------------------------------------------
# Schéma d'entrée : les 215 variables du modèle
# ---------------------------------------------------------------------------

# Champs obligatoires : toujours renseignés à l'entraînement (0 % de manquants).
# Sans eux, on refuse de scorer (sinon un client vide recevrait quand même un score).
REQUIRED_FIELDS = {
    "AMT_CREDIT": (float, Field(gt=0, description="Montant du crédit demandé (> 0)")),
    "AMT_ANNUITY": (float, Field(gt=0, description="Montant de l'annuité (> 0)")),
    "AMT_INCOME_TOTAL": (float, Field(gt=0, description="Revenu annuel du client (> 0)")),
    "DAYS_BIRTH": (
        float,
        Field(
            ge=-100 * 365,
            le=-18 * 365,
            description="Âge en jours, compté négativement depuis la demande (ex. -12000 ≈ 33 ans). Entre 18 et 100 ans.",
        ),
    ),
}

# Scores externes : facultatifs (souvent absents), mais normalisés entre 0 et 1 quand ils sont fournis
BOUNDED_FIELDS = {
    f"EXT_SOURCE_{i}": (Optional[float], Field(default=None, ge=0, le=1, description="Score externe normalisé (0 à 1)"))
    for i in (1, 2, 3)
}

# Toutes les autres variables : facultatives (null ou absente = valeur manquante, imputée par le modèle)
fields = {feature: (Optional[float], None) for feature in FEATURES}
fields.update(BOUNDED_FIELDS)
fields.update(REQUIRED_FIELDS)

# Modèle Pydantic construit automatiquement à partir de la liste des variables du modèle.
# extra="forbid" : une variable inconnue (ex. faute de frappe) renvoie une erreur au lieu d'être ignorée.
Client = create_model(
    "Client",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CLIENT]}),
    **fields,
)


class Prediction(BaseModel):
    """Réponse renvoyée par /predict."""

    probabilite_defaut: float = Field(description="Probabilité que le client ne rembourse pas (0 à 1)")
    decision: str = Field(description="'accorde' ou 'refuse'")
    seuil: float = Field(description="Seuil de décision : refus si probabilite_defaut >= seuil")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de scoring crédit - Prêt à Dépenser",
    description=(
        "Prédit la probabilité de défaut d'un client et la décision d'octroi du crédit.\n\n"
        "Erreurs possibles : **422** si les données sont invalides (champ obligatoire manquant, "
        "valeur hors plage, texte au lieu d'un nombre, variable inconnue), **500** si la prédiction échoue."
    ),
    version="1.0.0",
)


@app.get("/")
def accueil():
    """Page d'accueil : indique où trouver la documentation."""
    return {"message": "API de scoring crédit Prêt à Dépenser. Documentation : /docs"}


@app.get("/health")
def health():
    """Vérifie que l'API répond (utilisé pour contrôler le déploiement)."""
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_client(client: Client):
    """Score un client : les données sont validées par Pydantic avant d'arriver ici."""
    try:
        result = predict(model, [client.model_dump()])[0]
    except Exception as e:
        # Erreur inattendue pendant la prédiction : réponse 500 explicite plutôt qu'un crash de l'API
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    return {**result, "seuil": THRESHOLD}
