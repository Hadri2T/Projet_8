"""Tests unitaires du script d'inférence (src/predict.py), sans passer par l'API."""

import pandas as pd

from src.predict import THRESHOLD, get_features, imputer_rapide, predict


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


def test_imputation_rapide_identique_a_l_originale(model, client_exemple, client_minimal):
    # imputer_rapide() doit produire exactement le même tableau que l'imputation du Projet 6,
    # y compris dans les cas particuliers de chaque règle.
    # (Vérification complète sur les 356 251 clients du Projet 6 : python -m scripts.mesure_inference)
    cas = [
        client_exemple,
        client_minimal,  # 211 variables manquantes : règles 2 et 3 sur presque toutes les colonnes
        {**client_exemple, "FLAG_OWN_CAR": 0, "OWN_CAR_AGE": None},  # règle 1 : pas de voiture -> 0
        {**client_exemple, "FLAG_OWN_CAR": 1, "OWN_CAR_AGE": None},  # règle 1 : voiture -> médiane
        {**client_exemple, "INSTAL_PAYMENT_PERC_MEAN": float("inf")},  # division par 0 -> manquant
        {**client_exemple, "NAME_INCOME_TYPE_Working": 0, "EXT_SOURCE_1": None},  # autre groupe de profil
    ]
    X = pd.DataFrame(cas).reindex(columns=get_features(model)).astype(float)
    imputer = model.named_steps["imputer"]
    pd.testing.assert_frame_equal(imputer_rapide(imputer, X), imputer.transform(X))
