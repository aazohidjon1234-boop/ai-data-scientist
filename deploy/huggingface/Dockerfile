# Backend image for Hugging Face Spaces (free CPU tier: 2 vCPU / 16 GB RAM).
#
# Build context is the PROJECT ROOT:
#   docker build -f deploy/huggingface/Dockerfile .
#
# On a Space this file must sit at the repository root — `prepare-space.sh`
# assembles a ready-to-push directory with the correct layout.
FROM python:3.12-slim

# Spaces run the container as UID 1000; writing as root would fail at runtime.
RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY deploy/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

# Storage. The Space filesystem is EPHEMERAL on the free tier: uploads,
# reports and a SQLite database survive restarts only if DATABASE_URL points
# at an external database (see deploy/DEPLOY.md).
ENV DATA_ROOT=/data \
    UPLOAD_DIR=/data/datasets \
    REPORT_DIR=/data/reports \
    MODEL_DIR=/data/models \
    DATABASE_URL=sqlite:////data/app.db \
    PORT=7860

# Bundled example datasets live under UPLOAD_DIR/samples (see core/config.py).
COPY datasets/samples /data/datasets/samples
RUN mkdir -p /data/datasets/uploads /data/reports /data/models \
    && chown -R user:user /data /app

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','7860')}/api/health\")" || exit 1

# Shell form so ${PORT} is expanded — Spaces may override it.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
