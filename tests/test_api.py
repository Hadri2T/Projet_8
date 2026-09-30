"""Tests d'intégration de l'API (src/api.py) : requêtes HTTP de bout en bout, sans serveur."""

import pytest

import src.api

REQUIRED = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "DAYS_BIRTH"]


def champ_en_erreur(response):
    """Nom du champ signalé dans une réponse 422 (ex. 'AMT_CREDIT')."""
    return response.json()["detail"][0]["loc"][-1]


# ---------------------------------------------------------------------------
# Cas qui fonctionnent
# ---------------------------------------------------------------------------

def test_health(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_accueil(api):
    assert api.get("/").status_code == 200


def test_predict_client_exemple(api, client_exemple):
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 200
    assert response.json() == {"probabilite_defaut": 0.361, "decision": "accorde", "seuil": 0.5}


def test_predict_client_minimal(api, client_minimal):
    # Seuls les 4 champs obligatoires : les 211 autres sont imputés
    response = api.post("/predict", json=client_minimal)
    assert response.status_code == 200
    assert response.json()["decision"] == "refuse"


def test_valeurs_null_acceptees_sur_champs_facultatifs(api, client_minimal):
    client_minimal["EXT_SOURCE_1"] = None
    client_minimal["DAYS_EMPLOYED"] = None
    assert api.post("/predict", json=client_minimal).status_code == 200


# ---------------------------------------------------------------------------
# Données manquantes sur des champs obligatoires -> 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("champ", REQUIRED)
def test_champ_obligatoire_absent(api, client_exemple, champ):
    del client_exemple[champ]
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 422
    assert champ_en_erreur(response) == champ


@pytest.mark.parametrize("champ", REQUIRED)
def test_champ_obligatoire_a_null(api, client_exemple, champ):
    client_exemple[champ] = None
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 422
    assert champ_en_erreur(response) == champ


def test_client_vide(api):
    # Sans validation, un client vide recevrait quand même un score (0.34) : on doit le refuser
    response = api.post("/predict", json={})
    assert response.status_code == 422
    assert {e["loc"][-1] for e in response.json()["detail"]} == set(REQUIRED)


# ---------------------------------------------------------------------------
# Valeurs hors des plages attendues -> 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("DAYS_BIRTH", 12000),         # âge positif (les jours sont comptés négativement)
        ("DAYS_BIRTH", -5 * 365),      # 5 ans : trop jeune
        ("DAYS_BIRTH", -120 * 365),    # 120 ans : trop vieux
        ("AMT_INCOME_TOTAL", 0),       # revenu nul
        ("AMT_CREDIT", -1000),         # crédit négatif
        ("AMT_ANNUITY", 0),            # annuité nulle
        ("EXT_SOURCE_2", 3.5),         # score externe > 1
        ("EXT_SOURCE_1", -0.1),        # score externe < 0
    ],
)
def test_valeur_hors_plage(api, client_exemple, champ, valeur):
    client_exemple[champ] = valeur
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 422
    assert champ_en_erreur(response) == champ


# ---------------------------------------------------------------------------
# Types incorrects et requêtes mal formées -> 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("champ", ["AMT_CREDIT", "DAYS_EMPLOYED"])
def test_texte_au_lieu_d_un_nombre(api, client_exemple, champ):
    # Testé sur un champ obligatoire et sur un champ facultatif
    client_exemple[champ] = "beaucoup"
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 422
    assert champ_en_erreur(response) == champ


def test_variable_inconnue(api, client_exemple):
    # Une faute de frappe ne doit pas être ignorée en silence
    client_exemple["AMT_CREDITT"] = 1000
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 422
    assert champ_en_erreur(response) == "AMT_CREDITT"


def test_json_mal_forme(api):
    response = api.post("/predict", content="{pas du json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Erreur inattendue du modèle -> 500 explicite
# ---------------------------------------------------------------------------

def test_erreur_du_modele_renvoie_500(api, client_exemple, monkeypatch):
    # On remplace temporairement la fonction de prédiction par une fonction qui plante
    def predict_qui_plante(model, clients):
        raise ValueError("modèle indisponible")

    monkeypatch.setattr(src.api, "predict", predict_qui_plante)
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 500
    assert "modèle indisponible" in response.json()["detail"]
