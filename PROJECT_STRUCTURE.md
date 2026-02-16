# Project Structure

```
llm/
├── README.md                       # Main project overview
├── DEPLOYMENT_GUIDE.md             # Detailed deployment instructions
├── QUICK_START.md                  # Quick start guide (token setup)
├── API_GUIDE.md                    # REST API usage examples
├── docker-compose.yaml             # Infrastructure orchestration
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (secrets)
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── start.sh                        # Linux/WSL startup script
├── start.bat                       # Windows startup script
│
├── config/
│   └── lite-llm-config.yaml       # LiteLLM routing configuration
│
└── src/
    ├── main.py                     # Basic agent example
    ├── embeddings.py               # Self-hosted embedding service
    ├── rag_pipeline.py             # Simple RAG implementation
    ├── vector_store.py             # Qdrant vector database integration
    ├── production_rag.py           # Production RAG with vector DB
    ├── orchestrator.py             # Multi-agent orchestration
    ├── api.py                      # FastAPI REST API
    └── health_check.py             # Service health monitoring
```

## File Descriptions

### Configuration Files

- **`.env`**: Contains all secrets and configuration
  - Hugging Face token (required)
  - LiteLLM API key
  - FastAPI authentication key

- **`docker-compose.yaml`**: Defines the infrastructure
  - vLLM service (LLM inference)
  - TEI service (embeddings)
  - LiteLLM service (gateway)

- **`config/lite-llm-config.yaml`**: Routes API calls to internal services

### Documentation

- **`README.md`**: Project overview and architecture
- **`DEPLOYMENT_GUIDE.md`**: Step-by-step deployment instructions
- **`QUICK_START.md`**: Minimal setup guide (just the essentials)
- **`API_GUIDE.md`**: REST API usage with code examples

### Application Code

#### Basic Examples
- **`src/main.py`**: Simple agent using PydanticAI
- **`src/embeddings.py`**: Embedding service wrapper
- **`src/rag_pipeline.py`**: In-memory RAG system

#### Production Components
- **`src/vector_store.py`**: Qdrant vector database integration
- **`src/production_rag.py`**: Production RAG with persistence
- **`src/orchestrator.py`**: Multi-agent workflows
- **`src/api.py`**: REST API for external access

#### Utilities
- **`src/health_check.py`**: Verify all services are running

## What Each Component Does

### Infrastructure Layer (Docker)

```
┌─────────────────────────────────────────┐
│         Docker Compose                  │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │  vLLM    │  │   TEI    │  │LiteLLM ││
│  │(DeepSeek)│  │ (BGE-M3) │  │(Gateway)││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

### Application Layer (Python)

```
┌─────────────────────────────────────────┐
│         FastAPI REST API                │
│         (src/api.py)                    │
├─────────────────────────────────────────┤
│  ┌─────────────────┐  ┌───────────────┐│
│  │  Production RAG │  │ Orchestrator  ││
│  │  (vector store) │  │ (multi-agent) ││
│  └─────────────────┘  └───────────────┘│
├─────────────────────────────────────────┤
│  ┌─────────────────┐  ┌───────────────┐│
│  │  Vector Store   │  │  Embeddings   ││
│  │   (Qdrant)      │  │   (TEI)       ││
│  └─────────────────┘  └───────────────┘│
└─────────────────────────────────────────┘
```

## Usage Patterns

### 1. Simple Agent (Learning)
```bash
python src/main.py
```
Use this to understand how agents work.

### 2. RAG Pipeline (Development)
```bash
python src/rag_pipeline.py
```
In-memory RAG for testing and development.

### 3. Production RAG (Production)
```bash
python src/production_rag.py
```
Full RAG with vector database persistence.

### 4. Multi-Agent Workflows (Advanced)
```bash
python src/orchestrator.py
```
Complex workflows with specialized agents.

### 5. REST API (Production Deployment)
```bash
python src/api.py
```
Expose all functionality via REST API.

## Development Workflow

1. **Start Infrastructure**
   ```bash
   docker-compose up -d
   ```

2. **Verify Health**
   ```bash
   python src/health_check.py
   ```

3. **Test Components**
   ```bash
   # Test embeddings
   python src/embeddings.py
   
   # Test RAG
   python src/production_rag.py
   
   # Test orchestrator
   python src/orchestrator.py
   ```

4. **Start API**
   ```bash
   python src/api.py
   ```

5. **Integrate with Your Application**
   - Use the REST API endpoints
   - Or import Python modules directly

## Customization Points

### Change the LLM Model
Edit `docker-compose.yaml`, line ~27:
```yaml
--model deepseek-ai/DeepSeek-R1-Distill-Llama-70B
```

### Change the Embedding Model
Edit `docker-compose.yaml`, line ~42:
```yaml
command: --model-id BAAI/bge-m3
```

### Add Custom Agents
Edit `src/orchestrator.py` to add new specialized agents.

### Modify API Endpoints
Edit `src/api.py` to add new endpoints or change authentication.

## Security Considerations

### Secrets Management
All secrets are in `.env`:
- Never commit `.env` to git (it's in `.gitignore`)
- Use strong random keys for production
- Rotate keys regularly

### API Security
- Change `API_KEY` in `.env` for production
- Use HTTPS in production (add nginx/Traefik)
- Implement rate limiting
- Add IP whitelisting if needed

### Network Security
- Keep services on internal network
- Only expose LiteLLM and FastAPI externally
- Use firewall rules
- Monitor access logs

## Monitoring & Logging

### Check Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vllm
docker-compose logs -f tei
docker-compose logs -f litellm
```

### Monitor GPU Usage
```bash
nvidia-smi -l 1
```

### API Logs
FastAPI automatically logs all requests. For production, integrate with:
- Prometheus (metrics)
- Grafana (visualization)
- ELK Stack (log aggregation)

## Scaling

### Horizontal Scaling
- Add more vLLM instances behind a load balancer
- Use Redis for caching
- Deploy Qdrant cluster for vector storage

### Vertical Scaling
- Add more GPUs to vLLM
- Increase `--tensor-parallel-size`
- Allocate more RAM for vector database

## Next Steps

1. **Complete the setup** (see `DEPLOYMENT_GUIDE.md`)
2. **Test the examples** (run each Python file)
3. **Ingest your data** (use `production_rag.py` or API)
4. **Build your workflows** (customize `orchestrator.py`)
5. **Deploy to production** (use the REST API)

---

**Questions?** Review the documentation files or check the inline code comments.
