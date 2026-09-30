# Prêt à Dépenser — Mise en production du modèle de scoring crédit

Mise en production du modèle de scoring développé et versionné avec MLflow lors du projet précédent (*Initiez-vous au MLOps, partie 1*). Le modèle prédit la probabilité qu'un client ne rembourse pas son crédit, puis décide d'accorder ou de refuser la demande.

Livrables prévus : API de prédiction, tests automatisés, conteneurisation Docker, pipeline CI/CD, stockage des données de production, analyse du data drift, dashboard de monitoring et optimisation du temps d'inférence.

## Structure du dépôt

```
├── model/              # Modèle champion exporté du model registry MLflow
├── src/
│   └── predict.py      # Script d'inférence (chargement du modèle + prédiction)
├── notebooks/
│   └── 01_entrainement_modele_projet6.ipynb   # Notebook d'entraînement d'origine (Projet 6)
├── examples/
│   └── client_exemple.json   # Exemple de client au format attendu
└── requirements.txt
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

## Faire une prédiction

```bash
python -m src.predict examples/client_exemple.json
# {'probabilite_defaut': 0.361, 'decision': 'accorde'}
```

- `probabilite_defaut` : probabilité que le client ne rembourse pas (entre 0 et 1).
- `decision` : `refuse` si la probabilité est supérieure ou égale à 0.50, sinon `accorde`.

Le client est décrit dans un fichier JSON au format `{nom_variable: valeur}`. Une variable peut valoir `null` si elle est inconnue.
