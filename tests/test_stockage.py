"""Tests du stockage des appels (src/database.py et middleware de src/api.py) sur une base SQLite temporaire."""

import json

import anyio

import src.api


def test_appel_reussi_enregistre(api, client_exemple, derniere_ligne):
    api.post("/predict", json=client_exemple)
    ligne = derniere_ligne()
    assert ligne["statut_http"] == 200
    assert ligne["probabilite_defaut"] == 0.361
    assert ligne["decision"] == "accorde"
    assert ligne["erreur"] is None
    assert ligne["version"] == "local"  # pas de GIT_SHA en dehors de l'image Docker construite par la CI
    assert ligne["version_modele"] == "home_credit_scoring-v1"
    assert ligne["latence_ms"] > 0
    assert 0 < ligne["inference_ms"] <= ligne["latence_ms"]
    assert ligne["entrees"]["AMT_CREDIT"] == client_exemple["AMT_CREDIT"]


def test_seules_les_variables_renseignees_sont_stockees(api, client_minimal, derniere_ligne):
    api.post("/predict", json=client_minimal)
    assert derniere_ligne()["entrees"] == client_minimal


def test_erreur_422_enregistree(api, client_exemple, derniere_ligne):
    del client_exemple["AMT_CREDIT"]
    api.post("/predict", json=client_exemple)
    ligne = derniere_ligne()
    assert ligne["statut_http"] == 422
    assert "AMT_CREDIT" in ligne["erreur"]
    assert ligne["probabilite_defaut"] is None
    # Les erreurs portent aussi les deux versions : on sait quelle version du code les a produites
    assert ligne["version"] == "local"
    assert ligne["version_modele"] == "home_credit_scoring-v1"


def test_erreur_500_enregistree(api, client_exemple, derniere_ligne, monkeypatch):
    def predict_qui_plante(model, clients):
        raise RuntimeError("modèle indisponible")

    monkeypatch.setattr(src.api, "predict", predict_qui_plante)
    api.post("/predict", json=client_exemple)
    ligne = derniere_ligne()
    assert ligne["statut_http"] == 500
    assert "modèle indisponible" in ligne["erreur"]


def test_health_non_enregistre(api, nombre_de_lignes):
    # Seuls les appels à /predict sont stockés (les contrôles de santé de Render rempliraient la base)
    avant = nombre_de_lignes()
    api.get("/health")
    assert nombre_de_lignes() == avant


def test_base_en_panne_n_empeche_pas_de_repondre(api, client_exemple, monkeypatch):
    def ecriture_qui_plante(ligne):
        raise ConnectionError("base injoignable")

    monkeypatch.setattr(src.api, "enregistrer_appel", ecriture_qui_plante)
    response = api.post("/predict", json=client_exemple)
    assert response.status_code == 200
    assert response.json()["probabilite_defaut"] == 0.361


def test_reponse_envoyee_avant_l_ecriture_en_base(client_exemple, monkeypatch):
    """Le client n'attend pas la base : la réponse est entièrement envoyée AVANT l'écriture (tâche de fond).

    Le client de test de FastAPI attend la fin des tâches de fond avant de rendre la main : il ne permet
    pas de voir l'ordre. On appelle donc l'application directement (protocole ASGI), en notant dans quel
    ordre se produisent la fin de l'envoi de la réponse et l'écriture en base.
    """
    evenements = []
    monkeypatch.setattr(src.api, "enregistrer_appel", lambda ligne: evenements.append("écriture en base"))

    corps = json.dumps(client_exemple).encode()
    requete = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/predict", "raw_path": b"/predict", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json")], "client": ("test", 1), "server": ("test", 80),
    }

    async def recevoir():
        return {"type": "http.request", "body": corps, "more_body": False}

    async def envoyer(message):
        # Dernier morceau de la réponse : le client a tout reçu
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            evenements.append("réponse envoyée")

    anyio.run(src.api.app, requete, recevoir, envoyer)
    assert evenements == ["réponse envoyée", "écriture en base"]
