# 🗺️ Production Roadmap: Enterprise AI Platform

This document outlines the step-by-step plan to upgrade the current demo setup into a fully robust, secure, and scalable Enterprise AI Platform.

---

## 🏗️ Phase 1: Production Infrastructure

### 1.1 Dedicated AI Server Setup
- [ ] **Provision Hardware**:
    - Recommended: Ubuntu Server 22.04 LTS
    - CPU: 8+ Core modern CPU (AMD Ryzen 7/9 or Intel i7/i9)
    - GPU: **Not Required** (Optimized for CPU-only inference)
    - Storage: 100GB+ NVMe SSD
    - RAM: 32GB+ System RAM
- [ ] **OS Optimization**:
    - Install Docker Engine
    - Install **Ollama** (handled via Docker in current stack)
    - Configure `daemon.json` for proper log rotation

### 1.2 Production Docker Environment
- [x] **Update `docker-compose.yaml`**:
    - [x] Add `restart: always` pipelines
    - [x] Define explicit resource limits (CPU/RAM)
    - [x] Create named configuration for persistent volumes
    - [x] Create isolated internal network (`backend-net`)
- [x] **Container Isolation**:
    - [x] Ensure database (Qdrant) and Redis are not exposed publicly
    - [x] Only Nginx/Traefik should handle incoming HTTP/S traffic

### 1.3 Reverse Proxy & HTTPS
- [x] **Deploy Nginx/Traefik**:
    - [x] Set up a reverse proxy container
    - [x] Route traffic to LiteLLM (port 4000) and API (port 8888)
- [ ] **SSL Configuration**:
    - Generate certificates via Let's Encrypt (Certbot)
    - Enforce HTTPS redirect
- [ ] **Domain Setup**:
    - Configure internal DNS (e.g., `ai.company.internal`)

### 1.4 Monitoring & Logging stack
- [x] **Deploy Observability Stack**:
    - [x] **Prometheus**: Scrape metrics from vLLM and LiteLLM
    - [x] **Grafana**: Visualize GPU usage, token throughput, latency
    - [x] **Loki**: Centralized log aggregation for all Docker containers via Promtail.
- [ ] **Alerting**:
    - Set up alerts for GPU OOM, High Latency, Service Down

---

## 🧠 Phase 2: Models & Performance

### 2.1 Model Selection & optimization
- [x] **Select Models**:
    - [x] **Local General/Chat**: Qwen2.5-7B-Instruct (GGUF via Ollama)
    - [x] **Cloud Fallback**: DeepSeek-R1-70B (via Groq/OpenAI for high-reasoning)
- [x] **Optimization**:
    - [x] Implement **GGUF Quantization** (4-bit/5-bit) for CPU.
    - [x] Configure **Ollama** for concurrent CPU request handling.

### 2.2 Advanced Embeddings
- [x] **Upgrade Embedding Service**:
    - [x] deploy **BGE-M3** (already selected, verify utilization)
    - [x] Enable **Batch Processing** in TEI for higher throughput

---

## 🔐 Phase 3: Application Security

### 3.1 Authentication & RBAC
- [x] **Implement Identity Provider**:
    - [x] Implemented **OAuth2 with Password Grant**.
    - [x] Implemented **JWT (JSON Web Tokens)** for secure stateless sessions.
    - [x] Created `src/auth.py` for token handling and password hashing.
    - [x] Integrated `FAKE_USERS_DB` (ready for Postgres merge).
- [x] **Role-Based Access Control (RBAC)**:
    - [x] Initial API Key AND **Identity-based access** implemented.

### 3.2 Network Hardening
- [ ] **Firewall**:
    - [ ] Use `ufw` or cloud firewalls
- [x] **Internal Traffic**:
    - [x] Successfully isolated via Docker `llm-network` internal bridge.
    - [x] Nginx acts as the single point of entry with `X-API-Key` validation.

### 3.3 Secrets Management
- [ ] **Vault Integration**:
    - (Optional) HashiCorp Vault for key rotation
- [ ] **Environment Security**:
    - Ensure `.env` is never committed
    - Use Docker Secrets for sensitive runtime variables

---

## 🌐 Phase 4: Enterprise LiteLLM Gateway

### 4.1 Intelligent Routing
- [x] **Configure Header-Based Routing**:
    - [x] Example `coding-assistant` route configured in LiteLLM.
    - [ ] Add more specialized models as needed.
- [ ] **Load Balancing**:
    - Deploy multiple vLLM workers for the same model
    - Configure LiteLLM to load balance (Round Robin)

### 4.2 Application Resilience
- [ ] **Failover Config**:
    - Setup fallback models (e.g., if Main GPU fails -> Route to External API or smaller local model)
- [x] **Redis Caching**:
    - [x] Enable semantic caching in LiteLLM to reduce redundant computations

### 4.3 Analytics & Audit
- [x] **LiteLLM Database**:
    - [x] **Postgres** added to stack for persistent usage tracking.
    - [x] `success_callback` configured to log 모든 usage in database.
- [ ] **Cost Dashboards**:
    - [ ] Visualize usage per team/department in Grafana.

---

## 📦 Phase 5: Data & RAG Platform

### 5.1 Production Vector Database
- [x] **Deploy Qdrant Cluster**:
    - [x] Move from local single-node to distributed mode (if high scale)
    - Configure periodic snapshots/backups to S3/MinIO

### 5.2 Data Ingestion Pipelines
- [x] **Data Ingestion Pipeline**:
    - [x] Created `src/ingest.py` for automated indexing of PDFs, TXT, and MD files.
    - [x] Implemented chunking and metadata preservation.
    - [ ] Add support for SQL/Web content scraping.
- [ ] **Permission-Aware Indexing**:
    - Store ACLs (Access Control Lists) alongside vector embeddings
    - Filter search results based on User ID at query time

---

## 🤖 Phase 6: Agents & Developer Platform

### 6.1 Agent Orchestrator
- [x] **Agent Orchestrator**:
    - [x] Standardized on **PydanticAI** for structured multi-agent reasoning.
    - [x] Implemented **Tool-Calling** for agents to search the knowledge base.
- [x] **Specialized Agents**:
    - [x] Implemented `Legal Assistant` (Moroccan law context).
    - [x] Implemented `HR Assistant` (Internal policy context).
    - [x] Implemented `IT Support Agent` (Infrastructure context).

### 6.2 Tool Integrations
- [x] **Connect External Tools**:
    - [x] **Slack/Teams**: Automated notifications via `post_to_slack`.
    - [x] **Email**: Official document sharing via `send_email` tool.
    - [ ] **Jira/GitHub API**: Planned for next phase.

---

## 📈 Phase 7: DevOps & Scaling

### 7.1 CI/CD Pipelines
- [x] **Deployment Automation**:
    - [x] Created GitHub Actions workflow (`.github/workflows/deploy.yml`).
    - [x] Implemented automated health checks post-deployment.
- [ ] **Linting & Testing**:
    - [ ] Add Ruff/Black linting to pipeline.
    - [ ] Add Pytest for agents.

### 7.2 Developer Platform
- [x] **Unified API Gate**:
    - [x] Polished `src/api.py` with FastAPI.
    - [x] Implemented specialized agent endpoints (`/agent/legal`, `/agent/hr`, `/agent/it`).
    - [x] **OpenAPI/Swagger** (Interactive docs) auto-enabled at `/docs`.

### 7.3 High Availability (HA)
- [x] **Redundancy**:
    - [x] Implemented multiple replicas (2x) for `api` and `litellm` services.
    - [x] Nginx configured to load balance traffic across stateless instances.
- [x] **Auto-Scaling**:
    - [x] Health-checked and controlled via Docker Compose `deploy` policies.

---

## 🏛️ Phase 8: Enterprise Admin Dashboard

### 8.1 Administrative Control Center
- [ ] **Platform Management (Streamlit)**:
    - [ ] Build a centralized dashboard for platform owners.
    - [ ] Implemented **Knowledge Base Manager** (Index/Delete docs).
    - [ ] Implemented **User & Key Management** (CRUD operations).
    - [ ] Implemented **Live Analytics** (Token usage, latency, cost estimation).
- [ ] **Audit & Compliance**:
    - [ ] Exportable activity logs from Postgres.
    - [ ] Governance reports for executive stakeholders.

---

## 🗓️ Next Immediate Steps (Day 1)

1.  **Build Admin Dashboard**: Implement the Streamlit-based control center for non-developers.
2.  **User Management**: Integrate a formal Postgres User Table for RBAC.
3.  **Scale UP**: If latency increases, adjust the `replicas` count in `docker-compose.yaml`.

To restart the full stack with updates:
```bash
# Rebuild and restart
docker-compose up -d --build --remove-orphans
```
