"""Tests unitaires du script d'inférence (src/predict.py), sans passer par l'API."""

from src.predict import THRESHOLD, get_features, predict


def test_modele_attend_215_variables(model):
    assert len(get_features(model)) == 215


def test_prediction_client_exemple(model, client_exemple):
    # Valeur de référence : identique à la prédiction du modèle dans le notebook du Projet 6
    result = predict(model, [client_exemple])[0]
    assert result == {"probabilite_defaut": 0.361, "decision": "accorde"}


def test_probabilite_entre_0_et_1(model, client_exemple, client_minimal):
    for client in (client_exemple, client_minimal):
        proba = predict(model, [client])[0]["probabilite_defaut"]
        assert 0 <= proba <= 1


def test_decision_respecte_le_seuil(model, client_exemple, client_minimal):
    # client_exemple : 0.361 < 0.5 -> accorde ; client_minimal : 0.5052 >= 0.5 -> refuse
    for client in (client_exemple, client_minimal):
        result = predict(model, [client])[0]
        attendu = "refuse" if result["probabilite_defaut"] >= THRESHOLD else "accorde"
        assert result["decision"] == attendu
    assert predict(model, [client_minimal])[0]["decision"] == "refuse"


def test_variables_manquantes_imputees(model, client_minimal):
    # 211 variables absentes sur 215 : le modèle les impute au lieu de planter
    result = predict(model, [client_minimal])[0]
    assert result["probabilite_defaut"] == 0.5052


def test_plusieurs_clients_en_une_fois(model, client_exemple, client_minimal):
    results = predict(model, [client_exemple, client_minimal])
    assert len(results) == 2
    assert results[0] == predict(model, [client_exemple])[0]
