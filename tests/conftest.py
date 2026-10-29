"""Fixtures partagées par tous les tests (pytest les injecte automatiquement par leur nom)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Les tests écrivent dans une base SQLite temporaire, jamais dans la vraie base Neon.
# À définir AVANT d'importer l'API, car la connexion est créée à l'import de src.database.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/tests.db"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from src.api import app  # noqa: E402
from src.database import engine, predictions  # noqa: E402
from src.predict import load_model  # noqa: E402

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


@pytest.fixture
def derniere_ligne():
    """Fonction qui renvoie le dernier appel enregistré dans la table predictions."""
    def lire():
        with engine.connect() as conn:
            requete = select(predictions).order_by(predictions.c.id.desc()).limit(1)
            return conn.execute(requete).mappings().first()
    return lire


@pytest.fixture
def nombre_de_lignes():
    """Fonction qui renvoie le nombre d'appels enregistrés dans la table predictions."""
    def compter():
        with engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(predictions)).scalar()
    return compter
