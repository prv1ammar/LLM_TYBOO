FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and other necessary files
COPY src/ /app/src/
COPY .env /app/.env

# Expose the API port
EXPOSE 8888

# Command to run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8888"]
