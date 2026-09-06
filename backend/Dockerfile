# Build context is the PROJECT ROOT (see docker-compose.yml):
#   docker build -f backend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM links against the system libgomp (scikit-learn bundles its own).
# Without it `import lightgbm` fails and two models drop out of every run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
# bundled example datasets (also bind-mounted from the host in compose)
COPY datasets/samples ./datasets/samples

# storage locations (bind-mounted in docker-compose)
ENV UPLOAD_DIR=/app/datasets \
    REPORT_DIR=/app/reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
