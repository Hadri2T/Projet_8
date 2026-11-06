"""
Simulation de trafic de production : envoie des clients à l'API déployée, comme le feraient les
applications du département "Crédit Express". Chaque appel est enregistré par l'API dans la base Neon.

Les clients viennent du jeu de test Kaggle : 48 744 clients jamais vus par le modèle, sans TARGET
(on ne sait pas s'ils ont remboursé), exactement comme de vraies nouvelles demandes de crédit.

Deux vagues, envoyées l'une après l'autre :
- vague "normale" : des clients tels quels ;
- vague "derive"  : scénario d'une campagne qui attire des clients plus jeunes (< 35 ans) demandant
  des montants 30 % plus élevés. C'est un drift volontaire, pour vérifier que l'analyse le détecte.
Dans chaque vague, environ 2 % de requêtes invalides (champ oublié, valeur impossible, texte...).

Utilisation (depuis la racine du projet) :
    python scripts/simulation_trafic.py                                  # 2 000 + 1 000 clients vers Render
    python scripts/simulation_trafic.py --url http://127.0.0.1:8765 --normaux 20 --derives 10   # essai local
"""

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DONNEES = ROOT / "data" / "prepared_dataset.parquet"  # jeu préparé au Projet 6 (non versionné)
BILAN = ROOT / "data" / "simulation_vagues.json"  # dates de début et de fin de chaque vague
URL_RENDER = "https://scoring-api-latest-shx9.onrender.com"

# Les 215 variables attendues par l'API (mêmes noms que le client d'exemple)
with open(ROOT / "examples" / "client_exemple.json") as f:
    FEATURES = list(json.load(f))


def nettoyer(nom):
    """Même nettoyage des noms de colonnes qu'au Projet 6 (ex. 'Car loan' -> 'Car_loan')."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", nom)


def charger_clients(chemin=DONNEES):
    """Tous les clients du Projet 6 (entraînement + test Kaggle), avec les 215 variables du modèle et TARGET."""
    # On ne lit que les colonnes utiles : le fichier en contient 789
    colonnes = {nom: nettoyer(nom) for nom in pq.read_schema(chemin).names if nettoyer(nom) in FEATURES}
    return pd.read_parquet(chemin, columns=list(colonnes) + ["TARGET"]).rename(columns=colonnes)


def charger_clients_test(chemin=DONNEES):
    """Clients du jeu de test Kaggle (TARGET vide), avec les 215 variables du modèle."""
    clients = charger_clients(chemin)
    return clients[clients["TARGET"].isna()].drop(columns="TARGET")


def appliquer_derive(clients):
    """Scénario de dérive : clients de moins de 35 ans, montants demandés augmentés de 30 %.
    Les ratios calculés au Projet 6 à partir de ces montants sont recalculés pour rester cohérents."""
    jeunes = clients[clients["DAYS_BIRTH"] > -35 * 365].copy()
    for col in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        jeunes[col] *= 1.3
    jeunes["INCOME_CREDIT_PERC"] = jeunes["AMT_INCOME_TOTAL"] / jeunes["AMT_CREDIT"]
    jeunes["ANNUITY_INCOME_PERC"] = jeunes["AMT_ANNUITY"] / jeunes["AMT_INCOME_TOTAL"]
    # PAYMENT_RATE (annuité / crédit) ne change pas : les deux montants sont multipliés par 1,3
    return jeunes


def en_json(client):
    """Client -> dictionnaire envoyable en JSON, sans les valeurs manquantes (NaN n'existe pas en JSON)."""
    return {k: float(v) for k, v in client.items() if pd.notna(v)}


def corrompre(client):
    """Transforme un client en requête invalide, comme le ferait une application cliente mal codée."""
    client = dict(client)
    erreur = random.choice(["champ_oublie", "revenu_negatif", "age_impossible", "texte", "variable_inconnue"])
    if erreur == "champ_oublie":
        del client["AMT_CREDIT"]
    elif erreur == "revenu_negatif":
        client["AMT_INCOME_TOTAL"] = -client["AMT_INCOME_TOTAL"]
    elif erreur == "age_impossible":
        client["DAYS_BIRTH"] = 5 * 365  # positif : client né dans le futur
    elif erreur == "texte":
        client["AMT_CREDIT"] = "cinq cent mille"
    else:
        client["AMT_CREDITT"] = client["AMT_CREDIT"]  # faute de frappe dans le nom d'une variable
    return client


def envoyer(url, client):
    """POST /predict. Renvoie (code HTTP, réponse JSON ou None, durée vue par le client en secondes)."""
    requete = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(client).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    debut = time.perf_counter()
    try:
        # timeout large : l'offre gratuite de Render peut mettre ~1 min à se réveiller
        with urllib.request.urlopen(requete, timeout=120) as reponse:
            code, corps = reponse.status, json.loads(reponse.read())
    except urllib.error.HTTPError as e:  # l'API a répondu, mais avec une erreur (422, 500)
        code, corps = e.code, None
    except Exception:  # pas de réponse du tout (réseau coupé, délai dépassé)
        code, corps = None, None
    return code, corps, time.perf_counter() - debut


def lancer_vague(nom, url, clients, taux_erreur, paralleles):
    """Envoie une vague de clients (dont une part de requêtes invalides) et affiche son bilan."""
    requetes = [en_json(c) for c in clients.to_dict("records")]
    for i in random.sample(range(len(requetes)), round(len(requetes) * taux_erreur)):
        requetes[i] = corrompre(requetes[i])

    print(f"\n=== Vague '{nom}' : {len(requetes)} requêtes, {paralleles} en parallèle ===")
    debut = datetime.now(timezone.utc)
    resultats = []
    # Plusieurs requêtes en même temps, pour ne pas attendre le réseau entre chaque appel
    with ThreadPoolExecutor(max_workers=paralleles) as pool:
        for i, resultat in enumerate(pool.map(lambda c: envoyer(url, c), requetes), start=1):
            resultats.append(resultat)
            if i % 100 == 0 or i == len(requetes):
                ecoule = (datetime.now(timezone.utc) - debut).total_seconds()
                print(f"  {i}/{len(requetes)} envoyées ({ecoule:.0f} s, {i / ecoule:.1f} requêtes/s)", flush=True)
    fin = datetime.now(timezone.utc)

    statuts = Counter(str(code) for code, _, _ in resultats)
    decisions = Counter(corps["decision"] for _, corps, _ in resultats if corps)
    durees = pd.Series([d for _, _, d in resultats])
    bilan = {
        "vague": nom,
        "url": url,
        "debut": debut.isoformat(),
        "fin": fin.isoformat(),
        "requetes": len(requetes),
        "statuts": dict(statuts),
        "taux_refus": round(decisions["refuse"] / max(sum(decisions.values()), 1), 3),
        "duree_mediane_client_s": round(durees.median(), 2),
    }
    print(f"  Statuts HTTP : {dict(statuts)}")
    print(f"  Taux de refus : {bilan['taux_refus']:.1%} | durée médiane vue par le client : {bilan['duree_mediane_client_s']} s")
    return bilan


def main():
    parser = argparse.ArgumentParser(description="Simulation de trafic de production vers l'API de scoring")
    parser.add_argument("--url", default=URL_RENDER, help="adresse de l'API (défaut : Render)")
    parser.add_argument("--normaux", type=int, default=2000, help="nombre de clients de la vague normale")
    parser.add_argument("--derives", type=int, default=1000, help="nombre de clients de la vague de dérive")
    parser.add_argument("--taux-erreur", type=float, default=0.02, help="part de requêtes invalides")
    parser.add_argument("--paralleles", type=int, default=8, help="requêtes envoyées en même temps")
    args = parser.parse_args()

    random.seed(42)  # tirages reproductibles
    clients = charger_clients_test()
    normaux = clients.sample(args.normaux, random_state=42)
    # La vague de dérive est tirée parmi les clients restants : aucun client n'est envoyé deux fois
    derives = appliquer_derive(clients.drop(normaux.index)).sample(args.derives, random_state=42)
    print(f"{len(clients)} clients de test chargés | API : {args.url}")

    bilans = [
        lancer_vague("normale", args.url, normaux, args.taux_erreur, args.paralleles),
        lancer_vague("derive", args.url, derives, args.taux_erreur, args.paralleles),
    ]
    with open(BILAN, "w") as f:
        json.dump(bilans, f, indent=2, ensure_ascii=False)
    print(f"\nBilan enregistré dans {BILAN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
