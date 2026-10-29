# Image de l'API de scoring crédit
# Python 3.12 : même version que l'entraînement (le modèle est sérialisé avec cloudpickle, sensible à la version)
FROM python:3.12-slim

# Relie l'image publiée sur ghcr.io au dépôt GitHub (elle apparaît dans l'onglet "Packages" du dépôt)
LABEL org.opencontainers.image.source=https://github.com/Hadri2T/Projet_8

# libgomp1 : librairie de calcul parallèle (OpenMP) indispensable à LightGBM, absente de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (bonne pratique de sécurité : l'API n'a pas les droits administrateur)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Dépendances installées AVANT de copier le code : Docker garde cette étape en cache
# tant que requirements.txt ne change pas (builds beaucoup plus rapides)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Uniquement ce qui est nécessaire pour faire tourner l'API (pas de notebooks, pas de tests)
COPY --chown=user src/ src/
COPY --chown=user model/ model/
COPY --chown=user examples/ examples/

# Version du code (hash du commit), fournie par la CI au build et enregistrée avec chaque appel.
# Placée après les COPY : elle change à chaque build, sans invalider le cache des étapes précédentes.
ARG GIT_SHA=local
ENV GIT_SHA=$GIT_SHA

# Port par défaut. Un hébergeur peut imposer le sien via la variable PORT (Render utilise 10000) :
# la commande de lancement et le healthcheck lisent donc PORT au lieu d'un numéro écrit en dur
ENV PORT=7860
EXPOSE 7860

# Docker vérifie toutes les 30 s que l'API répond (le conteneur est marqué "unhealthy" sinon)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/health')"

# Lancement de l'API (le modèle est chargé une seule fois ici, au démarrage)
CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
