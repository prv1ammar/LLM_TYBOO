# 🛡️ Enterprise AI Platform (AGATE)

A production-ready, **CPU-optimized** AI infrastructure designed for Moroccan enterprises. AGATE provides a complete, self-hosted alternative to proprietary AI clouds, ensuring full data sovereignty and high performance without the need for expensive GPUs.

---

## 🏗️ Modern Hybrid Architecture

AGATE leverages a state-of-the-art hybrid stack that combines local efficiency with cloud-scale fallback:

```
┌─────────────────────────────────┐
│       Nginx Reverse Proxy       │ (Entry: Port 80, Security & HA)
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│     Streamlit Admin Dashboard   │ (Control: /admin/, Resource Mgmt)
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│    FastAPI Enterprise Logic     │ (Logic: /api/, Domain Agents)
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│       LiteLLM Gateway           │ (Routing: Local vs Cloud Fallback)
└───────────────┬─────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│    Ollama     │ │   Cloud API   │ (Groq/OpenAI Fallback)
│ (Qwen2.5-7B)  │ │ (DeepSeek-70B)│
└───────────────┘ └───────────────┘
```

---

## 🚀 Key Features

*   **💻 CPU-First Inference**: Powered by **Ollama** and **Qwen2.5-7B-Instruct (GGUF)**. No GPU required!
*   **🤖 Specialized Domain Agents**: Built-in assistants for **Legal (Moroccan Law)**, **HR (Internal Policy)**, and **IT Support**.
*   **📚 State-of-the-Art RAG**: Automated document ingestion pipeline indexing PDFs, TXT, and MD into **Qdrant Vector DB**.
*   **🌐 Admin Control Center**: A professional Streamlit UI to manage knowledge bases, track analytics, and manage users.
*   **🔐 Enterprise Security**: OAuth2/JWT Identity-based access + Nginx API-Key validation.
*   **📊 Full Observability**: Complete monitoring stack with **Prometheus, Grafana, and Loki** for logs and metrics.
*   **⚖️ Hybrid Model**: Automatic routing between local CPU models and high-reasoning cloud fallback (e.g., Groq).

---

## �️ Quick Start

### 1. Prerequisites
*   **Docker & Docker Compose**
*   **CPU**: 8+ Core modern processor.
*   **RAM**: 16GB+ (32GB recommended).

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env and set your JWT_SECRET_KEY and optional Cloud API Keys (GROQ_API_KEY)
```

### 3. Launch the Stack
```bash
docker-compose up -d --build
```

### 4. Initialize Local Model
```bash
# Pull the recommended Qwen2.5 model (first time only)
docker exec -it $(docker ps -qf "name=ollama") ollama pull qwen2.5:7b
```

---

## � Platform Access Points

| Service | Address | Role |
| :--- | :--- | :--- |
| **Admin Dashboard** | `http://localhost/admin/` | Control Center (Login: admin / password123) |
| **Enterprise API** | `http://localhost/api/docs` | Developer Documentation (Interactive) |
| **System Metrics** | `http://localhost/dashboard/` | Grafana Performance Monitoring |
| **Vector DB UI** | `http://localhost/qdrant/dashboard` | Inspect indexed documents |

---

## � Project Structure

*   `src/api.py`: The main FastAPI server orchestrating agents.
*   `src/dashboard.py`: Streamlit-based Administrative UI.
*   `src/ingest.py`: Automated Data Ingestion pipeline.
*   `src/auth.py`: Professional OAuth2/JWT identity layer.
*   `config/`: Infrastructure configurations (LiteLLM, Nginx, Prometheus).

---

## �️ Documentation & Guides
*   **[MANUAL_GUIDE.md](./MANUAL_GUIDE.md)**: Detailed step-by-step setup and usage instructions.
*   **[PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md)**: Strategic plan and completion status of platform features.

---

**Built for Moroccan enterprises 🇲🇦 | High Performance | Data Sovereign | Scalable**
