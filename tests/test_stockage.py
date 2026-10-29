"""Tests du stockage des appels (src/database.py et middleware de src/api.py) sur une base SQLite temporaire."""

import src.api


def test_appel_reussi_enregistre(api, client_exemple, derniere_ligne):
    api.post("/predict", json=client_exemple)
    ligne = derniere_ligne()
    assert ligne["statut_http"] == 200
    assert ligne["probabilite_defaut"] == 0.361
    assert ligne["decision"] == "accorde"
    assert ligne["erreur"] is None
    assert ligne["version"] == "local"  # pas de GIT_SHA en dehors de l'image Docker construite par la CI
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
