"""
Dashboard de monitoring de l'API de scoring (Streamlit).

Lancement (depuis la racine du projet) :
    streamlit run dashboard/app.py        -> http://localhost:8501

Principe : toutes les valeurs affichées sont calculées PAR LA BASE DE DONNÉES, avec des requêtes SQL
(PostgreSQL). Python envoie la requête et affiche le résultat. Sous chaque bloc, « Voir la requête SQL »
montre la requête exacte qui a été utilisée.

Deux bases au choix, dans le menu de gauche :
- Production (Neon) : les appels reçus par l'API déployée sur Render ;
- Local (Docker Compose) : la base PostgreSQL lancée par docker-compose.yml.
Le dashboard ne fait que LIRE les bases : il n'y écrit jamais.

Le dashboard sert au suivi rapide. L'analyse approfondie du data drift (Evidently, 215 variables)
reste dans notebooks/02_analyse_drift_monitoring.ipynb.
"""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import sqlalchemy as sa
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

# Les accès aux bases sont lus dans le fichier .env à la racine du projet (jamais publié sur GitHub)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_url_neon = os.getenv("DATABASE_URL")
_mdp_local = os.getenv("POSTGRES_PASSWORD")
BASES = {
    "Production (Neon)": _url_neon.replace("postgresql://", "postgresql+psycopg://", 1) if _url_neon else None,
    "Local (Docker Compose)": (
        f"postgresql+psycopg://scoring:{_mdp_local}@localhost:5433/scoring" if _mdp_local else None
    ),
}

# Seuils d'alerte : les mêmes que dans le notebook 02, fixés a priori
SEUIL_TAUX_ERREUR_PCT = 5
SEUIL_LATENCE_P95_MS = 1000
SEUIL_ECART_REFUS_POINTS = 10
SEUIL_DECISION = 0.5  # au-dessus, le crédit est refusé

# Référence = jeu de validation du projet précédent (61 502 clients jamais vus à l'entraînement),
# reconstruit avec le même découpage que le notebook 02. Valeurs calculées une fois pour toutes.
REFERENCE_TAUX_REFUS_PCT = 30.3
REFERENCE_SCORE_MOYEN = 0.387

# Variables suivies pour la dérive : libellé -> (variable envoyée à l'API, facteur, médiane de référence)
VARIABLES_DERIVE = {
    "Âge (ans)": ("DAYS_BIRTH", -1 / 365.25, 43.2),  # DAYS_BIRTH est un nombre de jours négatif
    "Montant du crédit": ("AMT_CREDIT", 1, 517_500),
    "PAYMENT_RATE (annuité / crédit)": ("PAYMENT_RATE", 1, 0.05),  # variable la plus importante du modèle
    "EXT_SOURCE_2 (score externe)": ("EXT_SOURCE_2", 1, 0.565),
}

# Une couleur fixe par version, dans l'ordre d'apparition des versions (palette testée pour les daltoniens)
PALETTE_CLAIR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PALETTE_SOMBRE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
GRIS = "#898781"  # seuils, références et libellés secondaires

# ---------------------------------------------------------------------------
# Requêtes SQL (PostgreSQL)
# ---------------------------------------------------------------------------

# Liste des versions de l'API présentes en base, de la plus ancienne à la plus récente,
# avec le ou les modèles qui ont servi chacune ("inconnue" si la version du modèle n'a pas été enregistrée)
SQL_VERSIONS = """
SELECT version,
       string_agg(DISTINCT coalesce(version_modele, 'inconnue'), ', ') AS version_modele,
       min(date_appel) AS premier_appel, count(*) AS appels
FROM predictions
GROUP BY version
ORDER BY premier_appel
"""

# Indicateurs clés, réutilisés par les deux requêtes suivantes.
# Les temps sont calculés sur les appels réussis (200) : une erreur 422 répond en 1 ms et fausserait la mesure.
INDICATEURS = """
    count(*)                                                    AS appels,
    (100.0 * avg((statut_http <> 200)::int))::float             AS taux_erreur_pct,
    count(*) FILTER (WHERE statut_http = 500)                   AS erreurs_500,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latence_ms)
        FILTER (WHERE statut_http = 200)                        AS latence_mediane_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latence_ms)
        FILTER (WHERE statut_http = 200)                        AS latence_p95_ms,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY inference_ms)
        FILTER (WHERE statut_http = 200)                        AS inference_mediane_ms,
    avg(probabilite_defaut)                                     AS score_moyen,
    (100.0 * avg((decision = 'refuse')::int))::float            AS taux_refus_pct"""

# Toutes les versions sélectionnées ensemble
SQL_CHIFFRES_CLES = f"""
SELECT {INDICATEURS}
FROM predictions
WHERE version = ANY(:versions)
"""

# Une ligne par version : c'est ce tableau qui montre l'amélioration (ou la dégradation) d'une version à l'autre.
# La colonne du modèle indique si un changement vient du code de l'API ou du modèle.
SQL_PAR_VERSION = f"""
SELECT version,
       string_agg(DISTINCT coalesce(version_modele, 'inconnue'), ', ') AS version_modele,
       min(date_appel) AS premier_appel, {INDICATEURS}
FROM predictions
WHERE version = ANY(:versions)
GROUP BY version
ORDER BY premier_appel
"""

# Latence minute par minute. Les versions ont tourné à des dates différentes : pour les comparer
# sur un même graphique, on compte les minutes depuis le premier appel de chaque version.
SQL_LATENCE = """
WITH appels AS (
    SELECT version, latence_ms, inference_ms,
           floor(extract(epoch FROM date_appel - min(date_appel) OVER (PARTITION BY version)) / 60)::int AS minute
    FROM predictions
    WHERE version = ANY(:versions) AND statut_http = 200
)
SELECT version, minute, count(*) AS appels,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY latence_ms)   AS latence_mediane_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latence_ms)   AS latence_p95_ms,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY inference_ms) AS inference_mediane_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY inference_ms) AS inference_p95_ms
FROM appels
GROUP BY version, minute
ORDER BY version, minute
"""

# Distribution des scores : 20 tranches de 0,05, en % des clients scorés de chaque version
SQL_SCORES = """
SELECT version,
       ((least(width_bucket(probabilite_defaut, 0, 1, 20), 20) - 0.5) / 20)::float AS centre_tranche,
       (100.0 * count(*) / sum(count(*)) OVER (PARTITION BY version))::float     AS part_pct
FROM predictions
WHERE version = ANY(:versions) AND statut_http = 200
GROUP BY version, centre_tranche
ORDER BY version, centre_tranche
"""

# Dérive : médiane d'une variable d'entrée, minute par minute. Les entrées des clients sont stockées
# en JSON : l'opérateur ->> lit une variable, ::float la convertit en nombre.
SQL_DERIVE = """
WITH appels AS (
    SELECT version,
           (entrees ->> CAST(:variable AS text))::float * CAST(:facteur AS float) AS valeur,
           floor(extract(epoch FROM date_appel - min(date_appel) OVER (PARTITION BY version)) / 60)::int AS minute
    FROM predictions
    WHERE version = ANY(:versions) AND statut_http = 200
)
SELECT version, minute,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY valeur) AS mediane,
       count(valeur) AS clients_renseignes
FROM appels
GROUP BY version, minute
ORDER BY version, minute
"""

# Répartition des appels par code HTTP
SQL_CODES = """
SELECT statut_http, count(*) AS appels
FROM predictions
WHERE version = ANY(:versions)
GROUP BY statut_http
ORDER BY statut_http
"""

# Erreurs 422 par champ en cause. La colonne erreur contient par exemple
# "AMT_CREDIT: Field required; DAYS_BIRTH: ..." : on la découpe en un motif par champ.
SQL_ERREURS_PAR_CHAMP = """
SELECT version, split_part(motif, ':', 1) AS champ, count(*) AS erreurs
FROM predictions
CROSS JOIN LATERAL regexp_split_to_table(erreur, '; ') AS motif
WHERE version = ANY(:versions) AND statut_http = 422
GROUP BY version, champ
ORDER BY erreurs DESC
"""

# ---------------------------------------------------------------------------
# Accès à la base
# ---------------------------------------------------------------------------


@st.cache_resource
def moteur(nom_base):
    """Connexion à la base, créée une seule fois puis réutilisée."""
    # pool_pre_ping : Neon coupe les connexions quand la base se met en veille
    return sa.create_engine(BASES[nom_base], pool_pre_ping=True)


@st.cache_data(ttl=60, show_spinner="Requête SQL en cours…")
def lire(nom_base, requete, **parametres):
    """Exécute une requête SQL et renvoie le résultat. Gardé en mémoire 60 s pour ne pas surcharger la base."""
    with moteur(nom_base).connect() as connexion:
        return pd.read_sql(sa.text(requete), connexion, params=parametres)


# ---------------------------------------------------------------------------
# Petites fonctions d'affichage
# ---------------------------------------------------------------------------


def nombre(valeur, decimales=0, unite=""):
    """Nombre au format français : 4 405 ms, 30,3 %… (tiret si la valeur manque)."""
    if valeur is None or pd.isna(valeur):
        return "—"
    texte = f"{valeur:,.{decimales}f}".replace(",", " ").replace(".", ",")
    return f"{texte} {unite}".strip()


def etat(alerte, texte):
    """Texte et couleur affichés sous un chiffre clé : jamais la couleur seule, toujours le mot OK ou Alerte."""
    return (f"Alerte : {texte}", "red") if alerte else (f"OK : {texte}", "green")


def voir_sql(requete):
    """Bloc repliable qui affiche la requête SQL utilisée."""
    with st.expander("Voir la requête SQL", icon=":material/code:"):
        st.code(requete.strip(), language="sql")


def echelle_versions(versions):
    """Couleur fixe de chaque version : elle ne change pas quand on filtre les versions."""
    return alt.Scale(domain=list(versions), range=[COULEURS[v] for v in versions])


def ligne_repere(valeur, texte, axe="y"):
    """Trait pointillé gris (seuil ou référence) avec son libellé."""
    donnees = alt.Chart(pd.DataFrame({"valeur": [valeur], "texte": [texte]}))
    trait = donnees.mark_rule(color=GRIS, strokeDash=[4, 4]).encode(**{axe: "valeur:Q"})
    if axe == "y":
        # Libellé au bout à droite du trait, pour ne pas recouvrir le début des courbes
        libelle = donnees.mark_text(color=GRIS, align="right", dx=-4, dy=-7).encode(
            y="valeur:Q", x=alt.value("width"), text="texte:N"
        )
    else:
        libelle = donnees.mark_text(color=GRIS, align="left", dx=4, baseline="top").encode(
            x="valeur:Q", y=alt.value(0), text="texte:N"
        )
    return trait + libelle


def courbes_par_minute(df, colonne, titre_axe, format_valeur):
    """Une courbe par version, minute par minute, avec infobulle au survol des points."""
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64))
        .encode(
            x=alt.X("minute:Q", title="Minutes depuis le premier appel de la version"),
            y=alt.Y(f"{colonne}:Q", title=titre_axe),
            color=alt.Color("version:N", title="Version", scale=echelle_versions(versions)),
            tooltip=[
                alt.Tooltip("version:N", title="Version"),
                alt.Tooltip("minute:Q", title="Minute"),
                alt.Tooltip(f"{colonne}:Q", title=titre_axe, format=format_valeur),
            ],
        )
    )


# ---------------------------------------------------------------------------
# Menu de gauche : choix de la base et des versions
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Monitoring de l'API de scoring", page_icon=":material/monitoring:", layout="wide")

with st.sidebar:
    st.header("Paramètres")
    nom_base = st.segmented_control("Base de données", list(BASES), default="Production (Neon)", required=True)
    if st.button("Rafraîchir les données", icon=":material/refresh:", width="stretch"):
        lire.clear()
    st.caption("Les résultats des requêtes sont gardés en mémoire 60 secondes.")

if BASES[nom_base] is None:
    st.error(
        "Accès à cette base absent du fichier .env : DATABASE_URL pour Neon, POSTGRES_PASSWORD pour la base locale.",
        title="Base non configurée",
    )
    st.stop()

try:
    toutes_versions = lire(nom_base, SQL_VERSIONS)
except Exception as erreur:  # base éteinte, en veille ou table absente
    st.error(
        f"{type(erreur).__name__}. Pour la base locale, lancer d'abord `docker compose up -d`.",
        title="Base injoignable",
    )
    st.stop()

if toutes_versions.empty:
    st.info("Aucun appel enregistré pour le moment.", title="Base vide")
    st.stop()

# Couleur de chaque version, d'après son ordre d'apparition (thème clair ou sombre)
palette = PALETTE_SOMBRE if st.context.theme.type == "dark" else PALETTE_CLAIR
COULEURS = {v: palette[i] if i < len(palette) else GRIS for i, v in enumerate(toutes_versions["version"])}

with st.sidebar:
    versions = st.pills(
        "Versions de l'API",
        list(toutes_versions["version"]),
        selection_mode="multi",
        default=list(toutes_versions["version"]),
        help="Hash du commit qui a produit la réponse, enregistré avec chaque appel.",
    )

st.title("Monitoring de l'API de scoring")

if not versions:
    st.info("Sélectionnez au moins une version dans le menu de gauche.", title="Aucune version sélectionnée")
    st.stop()

# Modèle(s) qui ont servi les versions sélectionnées, lus dans la base
modeles = toutes_versions.loc[toutes_versions["version"].isin(versions), "version_modele"]
modeles = sorted({m.strip() for liste in modeles for m in liste.split(",")})
st.caption(
    f"Modèle : {', '.join(modeles)} (nom et version dans le registry MLflow). "
    f"Toutes les valeurs sont calculées en SQL par la base « {nom_base} »."
)

# ---------------------------------------------------------------------------
# 1. Chiffres clés et alertes
# ---------------------------------------------------------------------------

st.subheader("1. Chiffres clés et alertes")
c = lire(nom_base, SQL_CHIFFRES_CLES, versions=versions).iloc[0]
ecart_refus = c.taux_refus_pct - REFERENCE_TAUX_REFUS_PCT

# (libellé, valeur, (texte sous le chiffre, couleur), aide)
cartes = [
    ("Appels", nombre(c.appels), (f"{len(versions)} version(s)", "gray"), None),
    (
        "Taux d'erreur",
        nombre(c.taux_erreur_pct, 1, "%"),
        etat(c.taux_erreur_pct > SEUIL_TAUX_ERREUR_PCT, f"seuil {SEUIL_TAUX_ERREUR_PCT} %"),
        "Part des appels en erreur (422 + 500).",
    ),
    (
        "Erreurs internes (500)",
        nombre(c.erreurs_500),
        etat(c.erreurs_500 > 0, "seuil 0"),
        "Une 500 signale un bug de l'API ou du modèle : aucune n'est tolérée.",
    ),
    (
        "Latence p95",
        nombre(c.latence_p95_ms, 0, "ms"),
        etat(c.latence_p95_ms > SEUIL_LATENCE_P95_MS, f"seuil {nombre(SEUIL_LATENCE_P95_MS)} ms"),
        "95 % des réponses sont plus rapides que cette valeur (validation + modèle, côté serveur).",
    ),
    (
        "Temps d'inférence médian",
        nombre(c.inference_mediane_ms, 0, "ms"),
        ("modèle seul", "gray"),
        "Temps passé dans le modèle (imputation + LightGBM), sans la validation des entrées.",
    ),
    (
        "Taux de refus",
        nombre(c.taux_refus_pct, 1, "%"),
        etat(
            abs(ecart_refus) > SEUIL_ECART_REFUS_POINTS,
            f"{'+' if ecart_refus >= 0 else ''}{nombre(ecart_refus, 1)} pts / référence",
        ),
        f"Référence : {nombre(REFERENCE_TAUX_REFUS_PCT, 1)} % de refus sur le jeu de validation. "
        f"Alerte au-delà de {SEUIL_ECART_REFUS_POINTS} points d'écart.",
    ),
]
# Deux lignes de trois cartes : volume et erreurs, puis temps de réponse et décisions
for ligne in (cartes[:3], cartes[3:]):
    for colonne, (libelle, valeur, (texte, couleur), aide) in zip(st.columns(3), ligne):
        colonne.metric(libelle, valeur, texte, delta_color=couleur, delta_arrow="off", help=aide, border=True)
voir_sql(SQL_CHIFFRES_CLES)

# ---------------------------------------------------------------------------
# 2. Comparaison des versions
# ---------------------------------------------------------------------------

st.subheader("2. Comparaison des versions")
st.caption(
    "Une colonne par version de l'API, avec le modèle qui l'a servie : on y lit l'effet de chaque changement "
    "(code ou modèle) sur les mêmes indicateurs."
)
par_version = lire(nom_base, SQL_PAR_VERSION, versions=versions)
# Une colonne par version et une ligne par indicateur : le tableau reste lisible quel que soit le nombre
# de versions (avec une ligne par version, les 11 indicateurs débordaient de l'écran)
comparaison = pd.DataFrame(
    {
        "Modèle": par_version["version_modele"],
        "Premier appel (UTC)": par_version["premier_appel"].dt.strftime("%d/%m/%Y %H:%M"),
        "Appels": par_version["appels"].map(nombre),
        "Taux d'erreur": par_version["taux_erreur_pct"].map(lambda v: nombre(v, 1, "%")),
        "Erreurs internes (500)": par_version["erreurs_500"].map(nombre),
        "Latence médiane": par_version["latence_mediane_ms"].map(lambda v: nombre(v, 0, "ms")),
        "Latence p95": par_version["latence_p95_ms"].map(lambda v: nombre(v, 0, "ms")),
        "Inférence médiane": par_version["inference_mediane_ms"].map(lambda v: nombre(v, 0, "ms")),
        "Score moyen": par_version["score_moyen"].map(lambda v: nombre(v, 3)),
        "Taux de refus": par_version["taux_refus_pct"].map(lambda v: nombre(v, 1, "%")),
    }
).set_axis(par_version["version"]).T
comparaison.index.name = "Indicateur"
st.dataframe(comparaison, alt="Indicateurs clés, une colonne par version de l'API")
voir_sql(SQL_PAR_VERSION)

# ---------------------------------------------------------------------------
# 3. Latence et temps d'inférence dans le temps
# ---------------------------------------------------------------------------

st.subheader("3. Latence et temps d'inférence")
mesure = st.segmented_control(
    "Mesure affichée",
    ["Latence totale", "Temps d'inférence (modèle seul)"],
    default="Latence totale",
    required=True,
)
latence = lire(nom_base, SQL_LATENCE, versions=versions)
if mesure == "Latence totale":
    graphique = courbes_par_minute(latence, "latence_p95_ms", "Latence p95 (ms)", ",.0f") + ligne_repere(
        SEUIL_LATENCE_P95_MS, f"seuil d'alerte : {nombre(SEUIL_LATENCE_P95_MS)} ms"
    )
else:
    graphique = courbes_par_minute(latence, "inference_p95_ms", "Temps d'inférence p95 (ms)", ",.0f")
st.altair_chart(graphique, alt=f"{mesure} (p95) minute par minute, une courbe par version")
st.caption("p95 calculé minute par minute sur les appels réussis. Survolez un point pour voir sa valeur.")
voir_sql(SQL_LATENCE)

# ---------------------------------------------------------------------------
# 4. Distribution des scores
# ---------------------------------------------------------------------------

st.subheader("4. Distribution des scores")
scores = lire(nom_base, SQL_SCORES, versions=versions)
graphique = (
    alt.Chart(scores)
    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64))
    .encode(
        x=alt.X("centre_tranche:Q", title="Probabilité de défaut prédite", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("part_pct:Q", title="Part des clients scorés (%)"),
        color=alt.Color("version:N", title="Version", scale=echelle_versions(versions)),
        tooltip=[
            alt.Tooltip("version:N", title="Version"),
            alt.Tooltip("centre_tranche:Q", title="Tranche (centre)", format=".3f"),
            alt.Tooltip("part_pct:Q", title="Part (%)", format=".1f"),
        ],
    )
) + ligne_repere(SEUIL_DECISION, "seuil de décision : refus au-delà de 0,5", axe="x")
st.altair_chart(graphique, alt="Distribution des probabilités de défaut, une courbe par version")
st.caption(
    f"Référence (jeu de validation) : score moyen {nombre(REFERENCE_SCORE_MOYEN, 3)}, "
    f"{nombre(REFERENCE_TAUX_REFUS_PCT, 1)} % de refus. Le score moyen et le taux de refus de chaque version "
    "sont dans le tableau de comparaison."
)
voir_sql(SQL_SCORES)

# ---------------------------------------------------------------------------
# 5. Dérive des données d'entrée
# ---------------------------------------------------------------------------

st.subheader("5. Dérive des données d'entrée")
choix = st.segmented_control("Variable suivie", list(VARIABLES_DERIVE), default="Âge (ans)", required=True)
variable, facteur, reference = VARIABLES_DERIVE[choix]
derive = lire(nom_base, SQL_DERIVE, versions=versions, variable=variable, facteur=facteur)
decimales = 0 if reference >= 1000 else 1 if reference >= 10 else 3
graphique = courbes_par_minute(derive, "mediane", f"{choix}, médiane", f",.{decimales}f") + ligne_repere(
    reference, f"référence (validation) : {nombre(reference, decimales)}"
)
st.altair_chart(graphique, alt=f"Médiane de {choix} minute par minute, comparée à la référence")
st.caption(
    "Médiane minute par minute des clients reçus, comparée à la médiane du jeu de validation. "
    "Un écart durable signale une dérive : la population qui demande un crédit a changé. "
    "L'analyse complète, sur les 215 variables, est dans le notebook 02 (Evidently)."
)
voir_sql(SQL_DERIVE)

# ---------------------------------------------------------------------------
# 6. Erreurs
# ---------------------------------------------------------------------------

st.subheader("6. Erreurs")
colonne_codes, colonne_champs = st.columns([1, 2])
with colonne_codes:
    codes = lire(nom_base, SQL_CODES, versions=versions)
    codes["signification"] = codes["statut_http"].map(
        {200: "Succès", 422: "Données invalides", 500: "Erreur interne"}
    )
    st.dataframe(
        codes,
        hide_index=True,
        alt="Nombre d'appels par code HTTP",
        column_config={"statut_http": "Code HTTP", "appels": "Appels", "signification": "Signification"},
    )
    voir_sql(SQL_CODES)
with colonne_champs:
    champs = lire(nom_base, SQL_ERREURS_PAR_CHAMP, versions=versions)
    if champs.empty:
        st.info("Aucune erreur 422 sur les versions sélectionnées.")
    else:
        graphique = (
            alt.Chart(champs)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("sum(erreurs):Q", title="Erreurs 422"),
                y=alt.Y("champ:N", title="Champ en cause", sort="-x", axis=alt.Axis(labelLimit=250)),
                color=alt.Color("version:N", title="Version", scale=echelle_versions(versions)),
                tooltip=[
                    alt.Tooltip("champ:N", title="Champ"),
                    alt.Tooltip("version:N", title="Version"),
                    alt.Tooltip("sum(erreurs):Q", title="Erreurs"),
                ],
            )
        )
        st.altair_chart(graphique, alt="Erreurs 422 par champ en cause et par version")
    voir_sql(SQL_ERREURS_PAR_CHAMP)
