FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 at build time (optional — also downloads at first request)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

EXPOSE 8888

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]
