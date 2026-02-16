# Self-Hosted Enterprise AI Stack

A production-ready, self-hosted AI infrastructure for Moroccan enterprises. This system provides:
- **Cost-effective** LLM inference using vLLM
- **Multilingual** embeddings with TEI (BGE-M3)
- **Unified API** via LiteLLM proxy
- **Full data sovereignty** - nothing leaves your infrastructure

## 🏗️ Architecture

```
┌─────────────────┐
│  Your App       │
│  (PydanticAI)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LiteLLM       │  ← OpenAI-compatible API
│   (Gateway)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ vLLM   │ │  TEI   │
│ (LLM)  │ │ (Emb)  │
└────────┘ └────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Linux** (Ubuntu 22.04+ recommended) or WSL2
- **NVIDIA GPU** (A100/H100 or RTX 3090/4090)
- **Docker** with `nvidia-container-toolkit`
- **Hugging Face account** (for model downloads)

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Hugging Face token
# Get it from: https://huggingface.co/settings/tokens
```

### 2. Start Infrastructure

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Wait for models to download (first time only, ~140GB)
# vLLM: DeepSeek-R1-Distill-Llama-70B (~70GB)
# TEI: BGE-M3 (~2GB)
```

### 3. Verify Health

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run health check
python src/health_check.py
```

### 4. Test the System

```bash
# Test basic agent
python src/main.py

# Test embeddings
python src/embeddings.py

# Test RAG pipeline
python src/rag_pipeline.py
```

## 📁 Project Structure

```
llm/
├── docker-compose.yaml          # Infrastructure definition
├── config/
│   └── lite-llm-config.yaml    # LiteLLM routing config
├── src/
│   ├── main.py                 # Basic agent example
│   ├── embeddings.py           # Embedding service
│   ├── rag_pipeline.py         # RAG implementation
│   └── health_check.py         # Service health checker
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (secrets)
```

## 🔧 Configuration

### GPU Memory Optimization

If you have limited GPU memory, enable quantization in `docker-compose.yaml`:

```yaml
# For vLLM service, add to command:
--quantization awq
# or
--quantization gptq
```

This reduces memory usage by ~50% with minimal quality loss.

### Model Selection

To use different models, edit `docker-compose.yaml`:

**For LLM (vLLM):**
- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (current, reasoning-focused)
- `meta-llama/Llama-3.3-70B-Instruct` (balanced, stable)
- `Qwen/Qwen2.5-72B-Instruct` (multilingual)

**For Embeddings (TEI):**
- `BAAI/bge-m3` (current, multilingual)
- `nomic-ai/nomic-embed-text-v1.5` (English-focused)

## 💰 Cost Comparison

### Before (OpenAI API)
- GPT-4 Turbo: $10/1M tokens
- Embeddings: $0.13/1M tokens
- **Monthly cost for high-volume workflows: $5,000+**

### After (Self-Hosted)
- GPU Server: $500-1000/month
- Unlimited inference
- **Monthly cost: $500-1000 (fixed)**

**Break-even:** ~500M tokens/month

## 🔒 Security for Moroccan Enterprises

### Data Sovereignty
- All data stays in your infrastructure
- No external API calls
- Compliant with local data protection laws

### Production Deployment
For production, add SSL/TLS termination:

```yaml
# Add to docker-compose.yaml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./certs:/etc/nginx/certs
```

## 📊 Monitoring & Optimization

### Enable Prompt Caching (vLLM)
Automatically enabled! Repeated prompts are cached for faster responses.

### Rate Limiting (LiteLLM)
Add to `lite-llm-config.yaml`:

```yaml
general_settings:
  max_parallel_requests: 100
  max_budget: 1000  # Internal budget tracking
```

## 🛠️ Troubleshooting

### Services won't start
```bash
# Check Docker logs
docker-compose logs vllm
docker-compose logs tei

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory
- Enable quantization (see Configuration above)
- Reduce `--max-model-len` in vLLM command
- Use smaller model variant

### Slow inference
- Check GPU utilization: `nvidia-smi`
- Increase `--tensor-parallel-size` if you have multiple GPUs
- Enable prefix caching (already enabled by default)

## 📚 Next Steps

1. **Integrate with your workflows**: Replace OpenAI API calls with LiteLLM endpoint
2. **Add vector database**: Use Qdrant or Milvus for production RAG
3. **Fine-tune models**: Customize for your specific use cases
4. **Scale horizontally**: Add more GPU servers behind a load balancer

## 📖 Documentation

- [vLLM Docs](https://docs.vllm.ai/)
- [LiteLLM Docs](https://docs.litellm.ai/)
- [TEI Docs](https://huggingface.co/docs/text-embeddings-inference)
- [PydanticAI Docs](https://ai.pydantic.dev/)

## 🤝 Support

This is a production-grade architecture used by enterprises worldwide. For Moroccan-specific deployment support, consider:
- OVH Cloud (European data centers)
- Local data center partnerships
- On-premise deployment with enterprise GPUs

---

**Built for Moroccan enterprises 🇲🇦 | Self-hosted | Scalable | Secure**
