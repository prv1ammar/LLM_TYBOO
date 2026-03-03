# Using a smaller base image for faster pulls
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (rarely change)
RUN apt-get update && apt-get install -y \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies (changes when requirements.txt updates) a
# We separate this from code copying to utilize Docker Layer Caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 at build time (optional — also downloads at first request)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

EXPOSE 8888

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]
