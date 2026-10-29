# Prêt à Dépenser — Mise en production du modèle de scoring crédit

Mise en production du modèle de scoring développé et versionné avec MLflow lors du projet précédent (*Initiez-vous au MLOps, partie 1*). Le modèle prédit la probabilité qu'un client ne rembourse pas son crédit, puis décide d'accorder ou de refuser la demande.

**API en ligne** : https://scoring-api-latest-shx9.onrender.com/docs. Elle est hébergée sur l'offre gratuite de Render : après 15 minutes sans requête, la première réponse peut prendre environ une minute.

Livrables prévus : API de prédiction, tests automatisés, conteneurisation Docker, pipeline CI/CD, stockage des données de production, analyse du data drift, dashboard de monitoring et optimisation du temps d'inférence.

## Structure du dépôt

```
├── model/              # Modèle champion exporté du model registry MLflow
├── src/
│   ├── predict.py      # Script d'inférence (chargement du modèle + prédiction)
│   ├── api.py          # API FastAPI qui expose le modèle
│   └── database.py     # Stockage des appels en base (PostgreSQL / SQLite)
├── notebooks/
│   └── 01_entrainement_modele_projet6.ipynb   # Notebook d'entraînement d'origine (Projet 6)
├── examples/
│   └── client_exemple.json   # Exemple de client au format attendu
├── tests/
│   ├── test_predict.py # Tests unitaires du script d'inférence
│   ├── test_api.py     # Tests d'intégration de l'API (cas valides et cas d'erreur)
│   └── test_stockage.py # Tests du stockage des appels
├── .github/workflows/
│   └── ci-cd.yml       # Pipeline CI/CD : tests → image Docker → déploiement
├── Dockerfile            # Image de l'API (production)
├── requirements.txt      # Dépendances de l'API (production)
└── requirements-dev.txt  # + dépendances de test
```

## Le modèle

| Élément | Valeur |
|---|---|
| Algorithme | LightGBM (`class_weight='balanced'`, 200 arbres, 15 feuilles) |
| Pipeline | Imputation des valeurs manquantes (`HomeCreditImputer`) + LightGBM |
| Origine | Model registry MLflow : `home_credit_scoring`, version 1, alias `champion` |
| Entrées | 215 variables (données de la demande + historique agrégé des crédits passés) |
| AUC (validation) | 0.785 |
| Seuil de décision | 0.50, optimisé pour minimiser le coût métier `10 × FN + FP` |

Un faux négatif (mauvais payeur accepté) coûte 10 fois plus cher à la banque qu'un faux positif (bon client refusé). Le seuil est donc choisi en fonction de ce coût, et non de l'accuracy.

Les valeurs manquantes sont acceptées : le pipeline les impute lui-même (0 si pas d'historique externe, sinon médiane du profil revenu × statut familial).

## Installation

Python 3.12 est requis, c'est la version utilisée à l'entraînement.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn src.api:app --reload
```

Les appels à `/predict` sont alors enregistrés dans un fichier SQLite local, `logs_local.db`.

Pour les enregistrer dans la base PostgreSQL de production :
1. Créez un fichier `.env` à la racine, contenant la ligne `DATABASE_URL=postgresql://...`. Ce fichier n'est jamais versionné.
2. Lancez l'API avec :

```bash
pip install -r requirements-dev.txt   # contient python-dotenv, nécessaire pour --env-file
uvicorn src.api:app --reload --env-file .env
```

La documentation interactive (Swagger) est disponible sur http://127.0.0.1:8000/docs. Le client d'exemple y est pré-rempli : il suffit de cliquer sur *Try it out* puis *Execute*.

| Route | Méthode | Rôle |
|---|---|---|
| `/` | GET | Message d'accueil |
| `/health` | GET | Vérifie que l'API répond : `{"status": "ok"}` |
| `/predict` | POST | Score un client |

Exemple avec curl :

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d @examples/client_exemple.json
# {"probabilite_defaut":0.361,"decision":"accorde","seuil":0.5}
```

### Données attendues et gestion des erreurs

Le client est envoyé en JSON au format `{nom_variable: valeur}`, avec les 215 variables du modèle.

- **Champs obligatoires** : `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL` (strictement positifs) et `DAYS_BIRTH` (âge en jours compté négativement, entre 18 et 100 ans). Ces champs sont toujours renseignés dans les données d'entraînement. Sans eux, on refuse de scorer.
- **Champs facultatifs** : toutes les autres variables. Une variable absente ou à `null` est considérée comme manquante et imputée par le modèle. Les scores `EXT_SOURCE_1/2/3` doivent être compris entre 0 et 1.

| Code | Cas |
|---|---|
| 200 | Prédiction réussie |
| 422 | Données invalides : champ obligatoire manquant, valeur hors plage, texte au lieu d'un nombre, variable inconnue. Le message indique le champ en cause. |
| 500 | Erreur inattendue pendant la prédiction |

Exemple de réponse 422 quand `AMT_CREDIT` manque :

```json
{"detail": [{"type": "missing", "loc": ["body", "AMT_CREDIT"], "msg": "Field required", ...}]}
```

## Lancer l'API avec Docker

```bash
docker build -t scoring-api .
docker run -p 7860:7860 scoring-api                  # appels enregistrés en SQLite dans le conteneur
docker run -p 7860:7860 --env-file .env scoring-api  # appels enregistrés dans PostgreSQL (Neon)
```

L'API est alors disponible sur http://localhost:7860/docs.

L'image ne contient que ce qui est nécessaire pour faire tourner l'API : le code `src/`, le modèle `model/` et le client d'exemple. Les notebooks, les tests et les outils de test en sont exclus.
- **Python 3.12**, comme à l'entraînement.
- **`libgomp1`** est ajoutée, car LightGBM en a besoin.
- L'API tourne avec un **utilisateur sans droits administrateur**.
- Un **contrôle de santé** Docker appelle `/health` toutes les 30 secondes.

Mesures en local : environ 110 Mo de RAM et 12 à 15 ms par requête.

## Lancer les tests

```bash
pip install -r requirements-dev.txt
pytest --cov=src
```

Il y a 39 tests, avec 94 % de couverture du code. Ils écrivent dans une base SQLite temporaire, jamais dans la base de production.
- **Script d'inférence** : prédiction de référence sur le client d'exemple (0.361), probabilité entre 0 et 1, respect du seuil, imputation des variables manquantes, prédiction de plusieurs clients à la fois.
- **API, cas valides** : `/health`, client complet, client réduit aux champs obligatoires, valeurs `null` sur les champs facultatifs.
- **API, cas d'erreur** :
  - données manquantes : chaque champ obligatoire absent ou à `null`, client vide ;
  - valeurs hors plage : âge positif, trop jeune ou trop vieux, revenu, crédit ou annuité nuls ou négatifs, score externe hors de [0, 1] ;
  - types incorrects : texte au lieu d'un nombre, variable inconnue, JSON mal formé ;
  - erreur interne du modèle, qui doit renvoyer une 500 explicite.
- **Stockage** :
  - un appel réussi est enregistré avec les bonnes valeurs ;
  - seules les variables renseignées sont stockées ;
  - les erreurs 422 et 500 sont enregistrées ;
  - `/health` n'est pas enregistré ;
  - une base en panne n'empêche pas l'API de répondre.

## Stockage des données de production

Chaque appel à `/predict`, réussi ou en erreur, est enregistré dans une base **PostgreSQL hébergée sur [Neon](https://neon.tech)**, région Frankfurt. La base est séparée de l'API : les données survivent aux redéploiements et aux mises en veille de Render. Elle est aussi accessible depuis l'extérieur, ce qui permet d'analyser le drift et d'alimenter le dashboard en continu.

Table `predictions` :

| Colonne | Contenu |
|---|---|
| `date_appel` | Date et heure de l'appel (UTC) |
| `statut_http` | 200 (succès), 422 (données invalides) ou 500 (erreur interne) |
| `latence_ms` | Temps total pour produire la réponse : validation + modèle |
| `inference_ms` | Temps du modèle seul |
| `entrees` | Variables envoyées par le client, au format JSON. Seules celles renseignées sont stockées. |
| `probabilite_defaut`, `decision` | Sortie du modèle |
| `erreur` | Champs en cause pour une 422, message pour une 500 |
| `version` | Hash du commit qui a traité l'appel, inscrit dans l'image Docker par la CI |

- **Base choisie par `DATABASE_URL`** : sur Render, cette variable est rangée dans les variables d'environnement du service ; en local, elle est dans le fichier `.env`. Si elle est absente, l'API écrit dans un fichier SQLite local.
- **Robustesse** : si la base est injoignable, l'API répond quand même et signale le problème dans ses logs.
- **Logs structurés** : en plus de la base, chaque appel produit une ligne JSON dans les logs de l'API, visibles dans l'onglet *Logs* de Render. Cette ligne contient tout, sauf les variables du client.
- **RGPD** : aucun identifiant client n'est stocké, et les données restent dans l'Union européenne.
- **Volume** : environ 6 Ko par appel, dont 5,7 Ko pour les variables du client d'exemple. Sans les valeurs manquantes, on économise environ 30 %. L'offre gratuite de Neon (0,5 Go) permet donc de l'ordre de 80 000 appels.

## Pipeline CI/CD

Le pipeline GitHub Actions ([.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)) se lance à chaque push et à chaque pull request sur `main`. Il enchaîne trois jobs, et chacun ne démarre que si le précédent a réussi :

| Job | Rôle | Push sur `main` | Pull request |
|---|---|---|---|
| `test` | Lance les tests pytest. Le job échoue si un test échoue ou si la couverture passe sous 90 %. | ✅ | ✅ |
| `build` | Construit l'image Docker, la lance, vérifie que `/health` et `/predict` répondent, puis la publie sur GitHub Container Registry. | ✅ (avec publication) | ✅ (sans publication) |
| `deploy` | Demande à Render de déployer la nouvelle image. | ✅ | ❌ |

- **Image Docker** : `ghcr.io/hadri2t/scoring-api`. Chaque version porte le hash du commit, et `latest` désigne la dernière.
- **Hébergement** : l'API tourne sur [Render](https://render.com), dans l'offre gratuite, à l'adresse https://scoring-api-latest-shx9.onrender.com. Render lance directement l'image publiée sur ghcr.io.
- **On déploie exactement l'image testée** : Render ne reconstruit pas l'API à partir du code. Le job `deploy` lui indique l'étiquette précise à lancer, c'est-à-dire le hash du commit.
- **Secrets** : le seul secret est l'URL de déploiement de Render (`RENDER_DEPLOY_HOOK`), rangée dans les secrets du dépôt GitHub. La publication sur ghcr.io utilise le jeton fourni automatiquement par GitHub (`GITHUB_TOKEN`).
- **Limites de l'offre gratuite** : l'API se met en veille après 15 minutes sans requête. La requête suivante la réveille, en environ une minute.
- **Pourquoi pas Hugging Face Spaces** : depuis juillet 2026, Hugging Face fait payer les Spaces Docker. Les Spaces Gradio restent gratuits, mais ils ne lancent pas d'image Docker : ils réinstallent le code Python eux-mêmes.

## Faire une prédiction sans l'API

```bash
python -m src.predict examples/client_exemple.json
# {'probabilite_defaut': 0.361, 'decision': 'accorde'}
```

- `probabilite_defaut` : probabilité que le client ne rembourse pas (entre 0 et 1).
- `decision` : `refuse` si la probabilité est supérieure ou égale à 0.50, sinon `accorde`.

Le client est décrit dans un fichier JSON au format `{nom_variable: valeur}`. Une variable peut valoir `null` si elle est inconnue.
