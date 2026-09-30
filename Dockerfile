# Image de l'API de scoring crédit
# Python 3.12 : même version que l'entraînement (le modèle est sérialisé avec cloudpickle, sensible à la version)
FROM python:3.12-slim

# libgomp1 : librairie de calcul parallèle (OpenMP) indispensable à LightGBM, absente de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (bonne pratique de sécurité, et exigé par Hugging Face Spaces : uid 1000)
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

# Port attendu par Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

# Docker vérifie toutes les 30 s que l'API répond (le conteneur est marqué "unhealthy" sinon)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Lancement de l'API (le modèle est chargé une seule fois ici, au démarrage)
CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
