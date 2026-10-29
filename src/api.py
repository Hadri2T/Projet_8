"""
API de scoring crédit "Prêt à Dépenser" (FastAPI).

Lancement en local (la variable DATABASE_URL est lue dans le fichier .env) :
    uvicorn src.api:app --reload --env-file .env
Documentation interactive (Swagger) : http://127.0.0.1:8000/docs
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.concurrency import run_in_threadpool

from src.database import creer_table, enregistrer_appel
from src.predict import THRESHOLD, get_features, load_model, predict

# Logs structurés : une ligne JSON par appel, lisible en direct dans l'onglet Logs de Render
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scoring_api")

# Version du code (hash du commit), inscrite dans l'image Docker par la CI ; "local" sinon
VERSION = os.getenv("GIT_SHA", "local")[:7]

# Chargement du modèle UNE SEULE FOIS, au démarrage de l'API, puis réutilisé par toutes les requêtes
# (le recharger à chaque requête ralentirait fortement l'API)
model = load_model()
FEATURES = get_features(model)

# Création de la table de stockage au démarrage. Si la base est injoignable, l'API démarre quand même :
# la prédiction passe avant le monitoring.
try:
    creer_table()
except Exception as e:
    logger.warning(json.dumps({"alerte": "base de données injoignable au démarrage", "detail": str(e)}))

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
def predict_client(client: Client, request: Request):
    """Score un client : les données sont validées par Pydantic avant d'arriver ici."""
    donnees = client.model_dump()
    # Pour le stockage, on ne garde que les variables renseignées : les manquantes prendraient de la place pour rien
    entrees = {k: v for k, v in donnees.items() if v is not None}
    try:
        debut = time.perf_counter()
        result = predict(model, [donnees])[0]
        inference_ms = (time.perf_counter() - debut) * 1000
    except Exception as e:
        # Erreur inattendue pendant la prédiction : réponse 500 explicite plutôt qu'un crash de l'API
        request.state.log = {"entrees": entrees, "erreur": str(e)}
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
    # Informations transmises au middleware, qui écrit la ligne en base
    request.state.log = {"entrees": entrees, **result, "inference_ms": round(inference_ms, 2)}
    return {**result, "seuil": THRESHOLD}


# ---------------------------------------------------------------------------
# Stockage des appels (monitoring et data drift)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def erreur_de_validation(request: Request, exc: RequestValidationError):
    """Données invalides (422) : on note les champs en cause pour le stockage, puis on renvoie
    la réponse 422 standard de FastAPI (inchangée)."""
    request.state.log = {
        "erreur": "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()),
    }
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def enregistrer_les_appels(request: Request, call_next):
    """Entoure chaque requête : chronomètre /predict, puis enregistre l'appel en base (succès comme erreur)."""
    if request.url.path != "/predict":
        return await call_next(request)

    debut = time.perf_counter()
    response = await call_next(request)  # la route /predict (ou le gestionnaire d'erreur) s'exécute ici
    ligne = {
        "date_appel": datetime.now(timezone.utc),
        "statut_http": response.status_code,
        "latence_ms": round((time.perf_counter() - debut) * 1000, 2),
        "version": VERSION,
        **getattr(request.state, "log", {}),  # entrées, score, décision ou erreur, selon le cas
    }

    # Écriture en base dans un thread à part, pour ne pas bloquer les autres requêtes pendant l'écriture.
    # Si la base est en panne, on le signale dans les logs mais la réponse part quand même.
    debut_ecriture = time.perf_counter()
    try:
        await run_in_threadpool(enregistrer_appel, ligne)
    except Exception as e:
        logger.warning(json.dumps({"alerte": "écriture en base impossible", "detail": str(e)}))
    ecriture_ms = round((time.perf_counter() - debut_ecriture) * 1000, 2)

    # Log structuré (sans les 215 variables, trop volumineuses pour les logs)
    resume = {k: v for k, v in ligne.items() if k != "entrees"}
    logger.info(json.dumps({**resume, "ecriture_bdd_ms": ecriture_ms}, default=str))
    return response
