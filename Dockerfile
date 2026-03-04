# Using a smaller base image for faster pulls
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (rarely change)
RUN apt-get update && apt-get install -y \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
# Corrected path for root context
COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project to /app
COPY src .

# 4. Install the 'tython' SDK in editable mode (so it's available globally)
RUN pip install -e ./tython

EXPOSE 8888

# Change command to run from the root, pointing to src.api
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]
