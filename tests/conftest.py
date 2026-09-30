"""Fixtures partagées par tous les tests (pytest les injecte automatiquement par leur nom)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import load_model

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "client_exemple.json"


@pytest.fixture(scope="session")
def model():
    """Modèle chargé une seule fois pour toute la session de tests."""
    return load_model()


@pytest.fixture(scope="session")
def api():
    """Client HTTP de test : envoie des requêtes à l'API sans lancer de serveur."""
    return TestClient(app)


@pytest.fixture
def client_exemple():
    """Client bancaire réel et anonymisé (215 variables, dont 64 manquantes).
    Rechargé à chaque test : un test peut le modifier sans impacter les autres."""
    with open(EXAMPLE_PATH) as f:
        return json.load(f)


@pytest.fixture
def client_minimal():
    """Client avec uniquement les 4 champs obligatoires."""
    return {"AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "AMT_INCOME_TOTAL": 150000, "DAYS_BIRTH": -12000}
