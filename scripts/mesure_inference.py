"""
Mesure du temps d'inférence du modèle, en local, sans API ni réseau (étape 4 : optimisation).

Compare deux façons d'exécuter le même modèle (model.pkl, inchangé) :
- "avant" : le pipeline du Projet 6 tel quel (model.predict_proba), imputation colonne par colonne ;
- "après" : predict() de src/predict.py, avec imputer_rapide() (colonnes traitées par groupes).

Le script :
0. vérifie la non-régression : mêmes données imputées et mêmes scores sur tous les clients du Projet 6 ;
1. rejoue des clients du jeu de test Kaggle un par un, comme l'API, et mesure le temps d'une prédiction ;
2. répartit ce temps entre les étapes : préparation du tableau (pandas), imputation, LightGBM ;
3. détaille le temps par fonction avec cProfile, l'outil de profilage intégré à Python.

Utilisation (depuis la racine du projet) :
    python -m scripts.mesure_inference                # 1 000 clients
    python -m scripts.mesure_inference --clients 200
"""

import argparse
import cProfile
import pstats
import time
import warnings

import numpy as np
import pandas as pd

from scripts.simulation_trafic import charger_clients
from src.predict import get_features, imputer_rapide, load_model, predict

# Avertissement de pandas sans effet ici (idxmax sur un client sans type de revenu renseigné)
warnings.filterwarnings("ignore", category=FutureWarning)


def chronometrer(fonction, elements):
    """Appelle fonction(element) pour chaque élément et renvoie les durées en millisecondes."""
    durees = []
    for element in elements:
        debut = time.perf_counter()
        fonction(element)
        durees.append((time.perf_counter() - debut) * 1000)
    return np.array(durees)


def resume(durees):
    return f"médiane {np.median(durees):6.2f} ms | p95 {np.percentile(durees, 95):6.2f} ms"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clients", type=int, default=1000, help="nombre de clients rejoués un par un")
    args = parser.parse_args()

    model = load_model()
    features = get_features(model)
    imputer, clf = model.named_steps["imputer"], model.named_steps["clf"]
    tous = charger_clients()

    # 0. Non-régression sur tous les clients (entraînement + test), par paquets de 50 000 pour la mémoire
    X_tous = tous[features].astype(float)
    ecart_max = 0.0
    for i in range(0, len(X_tous), 50_000):
        X = X_tous.iloc[i:i + 50_000]
        impute = imputer_rapide(imputer, X)
        pd.testing.assert_frame_equal(impute, imputer.transform(X))  # identique, à la décimale près
        ecart = np.abs(clf.predict_proba(impute)[:, 1] - model.predict_proba(X)[:, 1]).max()
        ecart_max = max(ecart_max, ecart)
    print(f"\n=== Non-régression sur {len(X_tous)} clients du Projet 6 ===")
    print(f"Données imputées identiques ; écart maximal entre les scores avant/après : {ecart_max}")

    # Clients au format reçu par predict() dans l'API : les 215 variables, None pour une valeur manquante
    test = tous[tous["TARGET"].isna()].drop(columns="TARGET").sample(args.clients, random_state=42)
    clients = [{k: (None if pd.isna(v) else float(v)) for k, v in ligne.items()} for ligne in test.to_dict("records")]

    def tableau(client):
        """Étape 1 de predict() : le client sous forme de tableau pandas à 215 colonnes."""
        return pd.DataFrame([client]).reindex(columns=features).astype(float)

    # Échauffement : les premiers appels sont plus lents (chargement en mémoire), on ne les compte pas
    for client in clients[:20]:
        model.predict_proba(tableau(client))
        predict(model, [client])

    # 1. Prédiction complète, un client à la fois comme dans l'API
    avant = chronometrer(lambda c: model.predict_proba(tableau(c)), clients)
    apres = chronometrer(lambda c: predict(model, [c]), clients)

    # 2. Les étapes, chronométrées séparément sur les mêmes clients
    tableaux = []
    preparation = chronometrer(lambda c: tableaux.append(tableau(c)), clients)
    imputation_avant = chronometrer(imputer.transform, tableaux)
    imputes = []
    imputation_apres = chronometrer(lambda X: imputes.append(imputer_rapide(imputer, X)), tableaux)
    lightgbm = chronometrer(clf.predict_proba, imputes)

    print(f"\n=== Temps d'une prédiction ({args.clients} clients, un par un, en local) ===")
    print(f"Prédiction complète, avant     : {resume(avant)}")
    print(f"Prédiction complète, après     : {resume(apres)} | {np.median(avant) / np.median(apres):.1f} fois plus rapide")
    print("Détail :")
    for nom, durees in [
        ("  1. Préparation (pandas)", preparation),
        ("  2. Imputation, avant", imputation_avant),
        ("  2. Imputation, après", imputation_apres),
        ("  3. LightGBM", lightgbm),
    ]:
        print(f"{nom:<31}: {resume(durees)}")

    # Comparaison : les mêmes clients envoyés en un seul appel (traitement « par lot »)
    debut = time.perf_counter()
    predict(model, clients)
    par_lot = (time.perf_counter() - debut) * 1000
    print(f"\nEn un seul appel ({args.clients} clients) : {par_lot:.0f} ms, soit {par_lot / args.clients:.3f} ms par client")

    # 3. Détail par fonction de la version actuelle (200 clients suffisent) : les 15 plus coûteuses
    print("\n=== cProfile : fonctions les plus coûteuses de predict() (temps cumulé, 200 prédictions) ===")
    profil = cProfile.Profile()
    profil.enable()
    for client in clients[:200]:
        predict(model, [client])
    profil.disable()
    pstats.Stats(profil).strip_dirs().sort_stats("cumulative").print_stats(15)
