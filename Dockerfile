FROM python:3.10-slim

LABEL maintainer="MNEMOSYNE-MNS"
LABEL description="MNEMOSYNE-MNS Memory Ranking Service"

WORKDIR /app

# Runtime packages only; the model bundle and E5 directory are supplied at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY docker/server.py ./server.py

# Prepare the model bundle before building. This keeps private weights out of Git
# while making the image self-contained when a bundle is supplied.
ARG MODEL_BUNDLE=deploy/model_bundle
ARG E5_MODEL_DIR=deploy/e5_model
COPY ${MODEL_BUNDLE}/ ./runs/rank_hybrid_v1/
COPY ${E5_MODEL_DIR}/ ./models/multilingual-e5-small/

ENV MNEMOSYNE_E5_MODEL=/app/models/multilingual-e5-small
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONPATH=/app/scripts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
