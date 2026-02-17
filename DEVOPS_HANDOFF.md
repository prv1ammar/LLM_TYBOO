# 🛠️ DevOps Deployment & Operational Guide

This document provides technical instructions for the final deployment and long-term maintenance of the AGATE Enterprise AI Platform.

---

## 🏗️ 1. Architecture Overview (CPU-Optimized)

AGATE is designed to run efficiently on standard enterprise server hardware without a dedicated GPU.

*   **Inference Layer**: Ollama (Handling Qwen2.5-7B-Instruct GGUF).
*   **Embedding Layer**: TEI (CPU-latest) with BGE-M3.
*   **Gateway**: LiteLLM (Routing local/cloud fallback).
*   **Storage**: Qdrant (Vector), Postgres (Analytics), Redis (Cache).
*   **Observability**: Prometheus (Metrics), Loki (Logs), Grafana (Dashboard).
*   **Edge**: Nginx (Reverse Proxy & HA Load Balancer).

---

## 🖥️ 2. Infrastructure Requirements

| Resource | Recommended | Minimum |
| :--- | :--- | :--- |
| **CPU** | 16 Cores (Threadripper / Epic / Xeon) | 8 Cores |
| **RAM** | 64GB | 32GB |
| **Storage** | 200GB NVMe SSD | 100GB SSD |
| **OS** | Ubuntu 22.04 LTS | Any Docker-ready Linux |

---

## 🔐 3. Environment Configuration (.env)

Ensure the following variables are set in your production `.env` file:

```bash
# Security
JWT_SECRET_KEY=  # Generate a 64-char random string
NGINX_API_KEY=   # Master key for outer Nginx gateway validation

# Cloud Fallback (Optional but Recommended)
GROQ_API_KEY=    # For high-reasoning fallback to DeepSeek-70B
# or 
OPENAI_API_KEY=

# Vector Database
QDRANT_URL=http://qdrant:6333

# Application
API_BASE_URL=http://api:8888
```

---

## 🚀 4. Deployment Procedure

### Step 1: Initialize Services
```bash
docker-compose up -d --build --remove-orphans
```

### Step 2: Warm-up the Models
Ollama needs to pull the GGUF model once during the initial setup:
```bash
docker exec -it $(docker ps -qf "name=ollama") ollama pull qwen2.5:7b
```

### Step 3: Verify Persistence
Confirm that volume mounts are correctly linked to:
- `./data/qdrant`
- `./data/postgres`
- `./data/ollama`

---

## ⚖️ 5. Scaling & High Availability (HA)

The `docker-compose.yaml` is pre-configured with **2 replicas** for the stateless services:
*   `api` (The domain logic & agent orchestrator)
*   `litellm` (The gateway)

**How to Scale Up:**
If the request load increases, simply update the `replicas` count in `docker-compose.yaml` and run:
```bash
docker-compose up -d --scale api=4 --scale litellm=4
```
*Nginx will automatically detect the new IP addresses of the containers and load balance traffic.*

---

## 📈 6. Monitoring & Troubleshooting

### 📊 Real-time Metrics (Grafana)
- **URL**: `http://<server-ip>/dashboard/`
- **Dashboard**: "LLM Enterprise Overview" (Pre-loaded).
- **Key Metrics**: Request Latency (ms), Tokens/sec, Active Users.

### 📜 Centralized Logs (Loki)
Logs from all 10+ containers are shipped via Promtail to Loki.
- Go to Grafana -> **Explore** -> Select **Loki** datasource.
- Query: `{container_name="api"}` to see only agent logs.

---

## 🛡️ 7. Security Hardening

1.  **SSL/TLS**: Update `nginx.conf` and map your certs in `docker-compose.yaml`.
2.  **Auth DB**: Currently `FAKE_USERS_DB` is used in `src/api.py`. Before launch, migration to the `postgres` service is required for production user management.
3.  **Firewall**: Only Port 80 (or 443) should be exposed to the public internet. All other ports (4000, 6333, 3100) are isolated within the `llm-network`.

---

## 🏁 8. Service Discovery Table

| Internal Service | External Endpoint | Role |
| :--- | :--- | :--- |
| `api:8888` | `/api/docs` | Agent API & Swagger |
| `dashboard:8501` | `/admin/` | Streamlit Control Center |
| `litellm:4000` | `/v1/` | Raw LLM Gateway |
| `grafana:3000` | `/dashboard/` | Monitoring |
| `qdrant:6333` | `/qdrant/dashboard` | Vector Store Explorer |

```
