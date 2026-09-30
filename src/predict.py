"""
Script d'inférence du modèle de scoring crédit "Prêt à Dépenser".

Le modèle est le LightGBM "champion" du Projet 6, copié depuis le model registry MLflow
(nom : home_credit_scoring, version 1, run BEST_LightGBM_auc0.785_cost30290_thr0.50).
C'est un pipeline scikit-learn : imputation des valeurs manquantes (HomeCreditImputer) + LightGBM.

Utilisation en ligne de commande :
    python -m src.predict examples/client_exemple.json
"""

import json
import pickle
import sys
from pathlib import Path

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
    probas = model.predict_proba(X)[:, 1]
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
