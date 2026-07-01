# Multi-stage build: keep the final image lean by not shipping build tools.
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt


FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    HF_HOME=/app/data/hf_cache

COPY --from=builder /root/.local /root/.local

COPY app ./app
COPY data/catalog_processed.json ./data/catalog_processed.json
COPY data/catalog_raw.json ./data/catalog_raw.json
COPY data/sample_conversations ./data/sample_conversations
COPY scripts ./scripts

# Pre-build the vector index at build time so the first request after a cold
# start doesn't pay the embedding cost inside the 30s request timeout.
# This downloads BAAI/bge-small-en-v1.5 once during the image build.
RUN python scripts/ingest_catalog.py

EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
