"""
Script d'inférence du modèle de scoring crédit "Prêt à Dépenser".

Le modèle est le LightGBM "champion" du Projet 6, copié depuis le model registry MLflow
(nom : home_credit_scoring, version 1, run BEST_LightGBM_auc0.785_cost30290_thr0.50).
C'est un pipeline scikit-learn : imputation des valeurs manquantes (HomeCreditImputer) + LightGBM.
Pour la rapidité, predict() n'appelle pas le pipeline directement : l'imputation est refaite par
imputer_rapide() (mêmes règles, mêmes valeurs apprises, résultat identique), puis LightGBM.

Utilisation en ligne de commande :
    python -m src.predict examples/client_exemple.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Chemin du modèle, calculé depuis ce fichier pour fonctionner quel que soit le dossier de lancement
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

# Seuil de décision optimal trouvé au Projet 6 (minimisation du coût métier 10*FN + FP)
# probabilité de défaut >= seuil -> crédit refusé
THRESHOLD = 0.50


def load_model(path=MODEL_PATH):
    """Charge le modèle une seule fois (à appeler au démarrage, pas à chaque prédiction)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_features(model):
    """Liste des 215 colonnes attendues par le modèle, dans l'ordre de l'entraînement."""
    return list(model.named_steps["clf"].feature_name_)


def imputer_rapide(imputer, X):
    """
    Même imputation que HomeCreditImputer.transform() (Projet 6), mais calculée sur une matrice numpy.
    Le code d'origine modifie le tableau pandas colonne par colonne (environ 150 opérations par client,
    chacune avec un coût fixe) : c'était 80 % du temps d'inférence (mesuré avec scripts/mesure_inference.py).
    Les valeurs utilisées (médianes) sont celles apprises au Projet 6 : imputer = étape du modèle chargé.
    """
    colonnes = X.columns
    valeurs = X.to_numpy(dtype=float, copy=True)
    # Les ratios peuvent valoir +/-inf (division par 0) : traités comme des valeurs manquantes
    valeurs[np.isinf(valeurs)] = np.nan

    # Règle 1 : pas de voiture -> âge de la voiture = 0, sinon médiane des propriétaires
    age = colonnes.get_loc("OWN_CAR_AGE")
    sans_voiture = valeurs[:, colonnes.get_loc("FLAG_OWN_CAR")] == 0
    valeurs[sans_voiture & np.isnan(valeurs[:, age]), age] = 0
    valeurs[np.isnan(valeurs[:, age]), age] = imputer.own_car_age_median_

    # Règle 2 : pas d'historique externe -> 0 (143 colonnes)
    externes = colonnes.get_indexer(imputer.external_cols_)
    bloc = valeurs[:, externes]
    bloc[np.isnan(bloc)] = 0
    valeurs[:, externes] = bloc

    # Règle 3 : médiane du groupe de profil du client (type de revenu + situation familiale),
    # repli sur la médiane globale si le groupe est inconnu ou n'a pas de médiane (31 colonnes).
    # Le groupe est calculé par la fonction d'origine, sur un tableau construit en une seule fois.
    groupe = imputer._group_key(pd.DataFrame(valeurs, columns=colonnes))
    medianes = imputer.group_medians_.reindex(groupe.values)[imputer.remaining_cols_].to_numpy()
    globales = imputer.global_medians_[imputer.remaining_cols_].to_numpy()
    medianes = np.where(np.isnan(medianes), globales, medianes)
    restantes = colonnes.get_indexer(imputer.remaining_cols_)
    bloc = valeurs[:, restantes]
    manquant = np.isnan(bloc)
    bloc[manquant] = medianes[manquant]
    valeurs[:, restantes] = bloc

    # Sécurité, comme dans l'original : toute valeur encore manquante -> 0
    valeurs[np.isnan(valeurs)] = 0
    return pd.DataFrame(valeurs, columns=colonnes, index=X.index)


def predict(model, clients):
    """
    Prédit la probabilité de défaut pour un ou plusieurs clients.

    clients : liste de dictionnaires {nom_colonne: valeur}, un dictionnaire par client.
              Une colonne absente ou à None est considérée comme manquante (imputée par le modèle).
    Retourne une liste de dictionnaires {probabilite_defaut, decision}.
    """
    # reindex : remet les colonnes dans l'ordre attendu et crée les colonnes absentes (NaN)
    # astype(float) : types homogènes pour LightGBM (vérifié : prédictions identiques)
    X = pd.DataFrame(clients).reindex(columns=get_features(model)).astype(float)
    # Équivaut à model.predict_proba(X), en plus rapide (voir imputer_rapide)
    X = imputer_rapide(model.named_steps["imputer"], X)
    probas = model.named_steps["clf"].predict_proba(X)[:, 1]
    return [
        {
            "probabilite_defaut": round(float(p), 4),
            "decision": "refuse" if p >= THRESHOLD else "accorde",
        }
        for p in probas
    ]


if __name__ == "__main__":
    # Lit un client au format JSON et affiche la prédiction
    with open(sys.argv[1]) as f:
        client = json.load(f)
    model = load_model()
    print(predict(model, [client])[0])
