# Image unique pour tous les participants : tableau de bord, serveur et hôpitaux.
# Chaque conteneur joue un seul rôle (voir docker-compose.yml).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 : calcul parallèle utilisé par TenSEAL et PyTorch
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch version CPU (beaucoup plus légère que la version GPU)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY src ./src
COPY tableau ./tableau
COPY tests ./tests

# Un utilisateur sans droits d'administrateur
RUN useradd --create-home --uid 1000 fl && mkdir -p /app/results /cles && chown -R fl /app /cles
USER fl

EXPOSE 8080 5050
CMD ["python", "-m", "src.tableau_de_bord", "--port", "8080"]
