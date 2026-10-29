FROM python:3.10-slim

LABEL maintainer="MNEMOSYNE-MNS"
LABEL description="MNEMOSYNE-MNS Memory Ranking Service"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY runs/rank_hybrid_v1/ ./runs/rank_hybrid_v1/

# Pre-download E5 model to container image
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; \
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-small'); \
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); \
    print('E5 model cached successfully')"

# Copy API server
COPY docker/server.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
