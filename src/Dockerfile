# Stage 1: build
FROM python:3.11-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache gcc g++ musl-dev libffi-dev curl

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-alpine

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . .

# Expose port and define command
EXPOSE 8888
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]