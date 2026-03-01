# LLM_TYBOO — CPU Edition

> Self-Hosted AI Platform — 10 cores / 24GB RAM / 124GB Storage

---

## Stack CPU-Optimized — Dual Model

| Service | Model | RAM | Port | Usage |
|---------|-------|-----|------|-------|
| llm-14b | Qwen2.5-14B Q4_K_M | ~9GB | 8000 | RAG, analyse, code, complexe |
| llm-3b | Qwen2.5-3B Q4_K_M | ~2GB | 8001 | Chat simple, Q&A, résumés |
| BGE-M3 (sentence-transformers) | — | ~2GB | — | Embeddings |
| Qdrant | — | ~500MB | 6333 | Vector DB |
| LiteLLM | — | ~300MB | **4000** | Proxy OpenAI-compatible |
| FastAPI | — | ~500MB | 8888 | API REST |
| Streamlit | — | ~300MB | 8501 | Dashboard |
| **n8n** | — | ~400MB | **5678** | Automation |
| PostgreSQL + Prometheus + Grafana | — | ~700MB | — | Logs/Monitoring |
| **Total** | | **~15.7GB / 24GB ✅** | | |

### Routing automatique
Le router choisit le modèle selon la complexité de la requête :
- **3B** → salutations, questions courtes, Q&A simples (~8-12 tok/s)
- **14B** → analyse, contrats, code, RAG, questions longues (~3-5 tok/s)

---

## Deploy — 4 étapes

### 1. Cloner et configurer
```bash
git clone https://github.com/votre-compte/LLM_TYBOO.git
cd LLM_TYBOO
cp .env.example .env
nano .env   # Mettre SERVER_IP + changer les clés
```

### 2. Télécharger le modèle (une seule fois — 4.5GB)
```bash
docker compose run --rm model-downloader
```

### 3. Lancer tout
```bash
docker compose up -d
docker compose ps   # Vérifier que tout est healthy
```

### 4. Vérifier
```bash
cd src && python health_check.py
```

---

## ⭐ Credentials n8n / LangChain / LiteLLM node

Après deploy, utiliser ces credentials dans **n8n**, **LangChain**, ou tout client OpenAI-compatible :

```
Base URL : http://YOUR_SERVER_IP:4000/v1
API Key  : sk-tyboo-2025          ← valeur LITELLM_KEY dans .env
Model    : internal-llm
```

### Dans n8n — Configuration LiteLLM/OpenAI node :
1. Ouvrir n8n → `http://YOUR_SERVER_IP:5678`
2. Settings → Credentials → New → **OpenAI**
3. Remplir :
   - API Key : `sk-tyboo-2025`
   - Base URL : `http://litellm:4000/v1`  *(Docker interne)*
4. Dans le node AI : Model = `internal-llm`

### Dans LangChain (Python) :
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://YOUR_SERVER_IP:4000/v1",
    api_key="sk-tyboo-2025",
    model="internal-llm",
)
response = llm.invoke("Analyse ce contrat...")
```

---

## URLs après deploy

| Service | URL |
|---------|-----|
| n8n Dashboard | http://YOUR_IP:5678 |
| FastAPI Swagger | http://YOUR_IP:8888/docs |
| Streamlit Dashboard | http://YOUR_IP:8501 |
| LiteLLM Proxy | http://YOUR_IP:4000 |
| Qdrant Dashboard | http://YOUR_IP:6333/dashboard |
| Grafana | http://YOUR_IP:3000 |

---

## Ingestion de documents

```bash
cp /path/to/docs/*.pdf src/documents/
cd src && python ingest.py --dir documents --collection knowledge_base
python ingest.py --stats
```

---

*LLM_TYBOO — CPU Edition — 2025/2026*
