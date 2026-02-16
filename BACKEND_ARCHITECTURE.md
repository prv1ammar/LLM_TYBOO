# LLM Backend Service & SDK Documentation

This document describes the backend-oriented architecture that allows you to reuse LLM capabilities across multiple projects.

## 🏗️ Architecture

The system is designed as a central **Microservice** that exposes AI capabilities via a REST API. Client applications use a provided **Python SDK** to interact with the service.

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Client App 1   │      │   Client App 2   │      │   Client App 3  │
│  (Web App)      │      │   (Data Pipeline)│      │   (Automation)  │
└────────┬────────┘      └────────┬─────────┘      └────────┬────────┘
         │                        │                         │
         │   Uses SDK             │ Uses SDK                │ Uses SDK
         ▼                        ▼                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                       LLM Backend Service                          │
│                       (src/backend_api.py)                         │
├───────────────────────────────┬────────────────────────────────────┤
│  • Job Queue (Async)          │  • Multi-Collection RAG            │
│  • Embeddings API             │  • Agent Orchestration             │
└──────────────┬────────────────┴──────────────────┬─────────────────┘
               │                                   │
      ┌────────▼────────┐                 ┌────────▼────────┐
      │  Inference      │                 │  Vector DB      │
      │  (vLLM/TEI)     │                 │  (Qdrant)       │
      └─────────────────┘                 └─────────────────┘
```

---

## 📦 Python SDK (`sdk/llm_client.py`)

We provide a copy-pasteable SDK that you can drop into any Python project.

### Installation for Client Apps

Your client projects only need:
```bash
pip install requests python-dotenv
```

### Basic Usage

```python
from llm_client import LLMBackendClient

# Connect to your self-hosted backend
client = LLMBackendClient(
    api_url="http://localhost:8888",
    api_key="your-secure-api-key"
)

# 1. Simple Chat
response = client.chat("Analyze this data...")

# 2. Get Embeddings (for your own ML models)
vector = client.embed_single("Some text to embed")
```

---

## ⚡ Features

### 1. Asynchronous Job Processing
Offload long-running tasks like batch processing or document analysis so your client apps remain responsive.

```python
# Submit a job
job_id = client.submit_job(
    job_type="batch_embed",
    params={"texts": large_list_of_texts}
)

# Check status later
status = client.get_job_status(job_id)
if status['status'] == 'completed':
    print(status['result'])
```

### 2. Multi-Project RAG (Retrieval Augmented Generation)
Support separate knowledge bases for different projects using "Collections".

```python
# In Project A (HR App)
client.rag_ingest(hr_docs, collection="hr_policies")
answer = client.rag_query("Vacation policy?", collection="hr_policies")

# In Project B (Tech Support)
client.rag_ingest(tech_docs, collection="technical_docs")
fix = client.rag_query("How to reset router?", collection="technical_docs")
```

### 3. Agentic Workflows
Reuse complex reasoning chains.

```python
# Analyze a complex contract or business document
analysis = client.analyze_document(document_text)
print(analysis['sentiment'])
print(analysis['key_points'])
```

---

## 🚀 Running the Backend Service

The backend service replaces the simple `main.py` script.

### Start command
```bash
# Start the production backend API
python src/backend_api.py
```

The service runs on `http://localhost:8888` by default.

### API Configuration
Configure in `.env`:
```ini
API_KEY=your-secure-api-key-here
LITELLM_URL=http://localhost:4000
```
