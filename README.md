# Tython — AI Platform

> Self-Hosted AI Platform — CPU Edition  
> 10 cores / 24GB RAM / 124GB Storage

---

## Stack CPU-Optimized — Dual Model

| Service | Model | RAM | Port | Usage |
|---------|-------|-----|------|-------|
| llm-14b | Qwen2.5-14B Q4_K_M | ~9GB | 8000 | RAG, analysis, code, complex |
| llm-3b | Qwen2.5-3B Q4_K_M | ~2GB | 8001 | Chat, Q&A, summaries |
| BGE-M3 (sentence-transformers) | — | ~2GB | — | Embeddings |
| Qdrant | — | ~500MB | 6333 | Vector DB |
| LiteLLM | — | ~300MB | **4000** | OpenAI-compatible Proxy |
| FastAPI | — | ~500MB | 8888 | REST API |
| Streamlit | — | ~300MB | 8501 | Dashboard |
| **n8n** | — | ~400MB | **5678** | Automation |
| PostgreSQL + Prometheus + Grafana | — | ~700MB | — | Logs/Monitoring |
| **Total** | | **~15.7GB / 24GB ✅** | | |

### Automatic Routing
- **3B** → greetings, short questions, simple Q&A (~8-12 tok/s)
- **14B** → analysis, contracts, code, RAG, long questions (~3-5 tok/s)

---

## Deploy — 4 Steps

### 1. Clone and configure
```bash
git clone https://github.com/your-account/tython.git
cd tython
cp .env.example .env
nano .env   # Set SERVER_IP + change keys
```

### 2. Download the model (once — 4.5GB)
```bash
docker compose run --rm model-downloader
```

### 3. Start everything
```bash
docker compose up -d
docker compose ps
```

### 4. Verify
```bash
cd src && python health_check.py
```

---

## Python SDK

```bash
pip install -e ./src/sdk
```

```python
from tython import TythonClient

client = TythonClient(
    api_url="http://YOUR_SERVER_IP:8888",
    api_key="your-api-key"
)

answer = client.chat("What are the key obligations in this contract?")
result = client.rag_query("What is the refund policy?", collection="knowledge_base")
vectors = client.embed(["text 1", "text 2"])
```

---

## ⭐ LiteLLM / n8n / LangChain Credentials

```
Base URL : http://YOUR_SERVER_IP:4000/v1
API Key  : sk-tython-2025
Model    : internal-llm
```

### LangChain (Python)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://YOUR_SERVER_IP:4000/v1",
    api_key="sk-tython-2025",
    model="internal-llm",
)
```

---

## Service URLs

| Service | URL |
|---------|-----|
| n8n | http://YOUR_IP:5678 |
| FastAPI Swagger | http://YOUR_IP:8888/docs |
| Dashboard | http://YOUR_IP:8501 |
| LiteLLM | http://YOUR_IP:4000 |
| Qdrant | http://YOUR_IP:6333/dashboard |
| Grafana | http://YOUR_IP:3000 |

---

*Tython — AI Platform — 2025/2026*
