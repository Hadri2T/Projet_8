"""
Stockage des appels à l'API (données de production) dans une base SQL.

Chaque appel à /predict, réussi ou en erreur, devient une ligne de la table "predictions" :
entrées du client, score, décision, temps de réponse, erreur éventuelle et version du code.
Ces données servent ensuite au monitoring (latence, taux d'erreur) et à l'analyse du data drift.

La base est choisie par la variable d'environnement DATABASE_URL :
- Render et local : PostgreSQL hébergé sur Neon (adresse rangée dans les secrets, jamais dans le code) ;
- tests et CI : DATABASE_URL absente -> petit fichier SQLite local.
"""

import os

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./logs_local.db")
# Neon fournit une adresse "postgresql://..." : on indique à SQLAlchemy d'utiliser le pilote psycopg (version 3)
DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# pool_pre_ping : vérifie qu'une connexion est encore ouverte avant de s'en servir
# (Neon coupe les connexions quand la base se met en veille après quelques minutes sans activité)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("date_appel", DateTime(timezone=True), nullable=False),
    Column("statut_http", Integer, nullable=False),  # 200 = OK, 422 = données invalides, 500 = erreur interne
    Column("latence_ms", Float, nullable=False),  # temps total pour produire la réponse (validation + modèle)
    Column("inference_ms", Float),  # temps du modèle seul
    Column("entrees", JSON),  # variables envoyées par le client (uniquement celles renseignées)
    Column("probabilite_defaut", Float),
    Column("decision", String(10)),
    Column("erreur", Text),
    Column("version", String(40)),  # hash du commit qui a produit la réponse
)


def creer_table():
    """Crée la table si elle n'existe pas encore (sans effet si elle existe déjà)."""
    metadata.create_all(engine)


def enregistrer_appel(ligne):
    """Ajoute une ligne (dictionnaire colonne -> valeur) dans la table predictions."""
    with engine.begin() as conn:  # begin() : la ligne est validée (commit) à la sortie du bloc
        conn.execute(predictions.insert().values(**ligne))
