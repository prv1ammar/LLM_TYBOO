# Tython — AI Platform (Unified Edition)

> **Self-Hosted AI Platform — CPU Optimized**  
> 10 cores / 24GB RAM / 124GB Storage

---

## 🚀 Unified Architecture (DevOps-Ready)

We have merged `api.py` and `backend_api.py` into a single **Unified Entry Point** on port `8888`. This simplifies deployment and allows both the Dashboard (JWT) and external SDKs (API Key) to communicate with the same service.

| Service | Port | Auth Mode | Usage |
|---------|------|-----------|-------|
| **Unified API** | `8888` | JWT / API-Key | Dashboard, SDK, RAG, Admin |
| **LiteLLM Proxy** | `4000` | Static Key | OpenAI-compatible (n8n, LangChain) |
| **Dashboard** | `8501` | JWT (Login) | Admin UI, Chat Interface |
| **n8n** | `5678` | Basic Auth | Workflows & Automations |

---

## 📦 Python SDK Installation

The SDK is now organized as a standard package.

```bash
# Install in editable mode
pip install -e ./tython
```

### Quick Usage

```python
from tython import TythonClient

client = TythonClient(
    api_url="http://135.125.4.184:8888",
    api_key="your-api-key-from-env"
)

# 1. Chat (Routed automatically to 3B or 14B)
answer = client.chat("Summarize the key obligations in this contract.")

# 2. RAG (Knowledge Base Search)
result = client.rag_query("What is the refund policy?", collection="enterprise_kb")

# 3. Embeddings (BGE-M3 / 1024D)
vectors = client.embed(["text 1", "text 2"])
```

---

## ⭐ n8n / LangChain Integration

Use these credentials for any OpenAI-compatible tool:

```yaml
Base URL : http://YOUR_SERVER_IP:4000/v1
API Key  : sk-tyboo-2025 (or as set in .env)
Model    : internal-llm
```

---

## 🛠️ Deploy (Production)

1. **Configure**: Update `.env` with `SERVER_IP` and `API_KEY`.
2. **Build & Start**:
   ```bash
   docker compose up -d --build
   ```
3. **Verify**:
   ```bash
   python test_tyboo.py
   ```

---

*Tython — Enterprise AI Platform — 2026*

