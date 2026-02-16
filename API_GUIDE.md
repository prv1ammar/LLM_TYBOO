# Example API Usage Guide

This guide shows how to use the FastAPI REST API for the self-hosted AI stack.

## Starting the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python src/api.py
```

The API will be available at: `http://localhost:8888`

## Authentication

All endpoints (except `/health` and `/info`) require an API key.

Add the API key to your request headers:
```
X-API-Key: your-secure-api-key-change-this-in-production
```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8888/health
```

Response:
```json
{
  "status": "healthy",
  "service": "self-hosted-ai-api"
}
```

### 2. System Info

```bash
curl http://localhost:8888/info
```

### 3. RAG Query

Ask questions based on your knowledge base:

```bash
curl -X POST http://localhost:8888/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-change-this-in-production" \
  -d '{
    "question": "What renewable energy projects exist in Morocco?",
    "top_k": 3,
    "include_sources": true
  }'
```

Response:
```json
{
  "answer": "Morocco has several major renewable energy projects...",
  "sources": [
    {
      "text": "Document text...",
      "metadata": {"topic": "energy"},
      "relevance_score": 0.89
    }
  ]
}
```

### 4. Ingest Documents

Add documents to the knowledge base:

```bash
curl -X POST http://localhost:8888/rag/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-change-this-in-production" \
  -d '[
    {
      "text": "Morocco is investing in renewable energy...",
      "metadata": {"topic": "energy", "year": 2024}
    },
    {
      "text": "The banking sector in Morocco is growing...",
      "metadata": {"topic": "finance", "year": 2024}
    }
  ]'
```

Response:
```json
{
  "document_ids": ["uuid-1", "uuid-2"],
  "count": 2
}
```

### 5. Analyze Document

Get structured analysis of a business document:

```bash
curl -X POST http://localhost:8888/agent/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-change-this-in-production" \
  -d '{
    "document": "Our company is expanding operations in Casablanca..."
  }'
```

Response:
```json
{
  "summary": "Company expansion plan for Casablanca...",
  "key_points": ["Expansion", "Hiring", "Budget"],
  "sentiment": "positive",
  "confidence": 0.92
}
```

### 6. Generate Content

Generate professional business content:

```bash
curl -X POST http://localhost:8888/agent/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key-change-this-in-production" \
  -d '{
    "prompt": "Write a brief introduction for our AI services",
    "context": "Moroccan enterprise focusing on digital transformation"
  }'
```

Response:
```json
{
  "content": "Our AI services help Moroccan enterprises..."
}
```

## Python Client Example

```python
import requests

API_URL = "http://localhost:8888"
API_KEY = "your-secure-api-key-change-this-in-production"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Query RAG system
response = requests.post(
    f"{API_URL}/rag/query",
    headers=headers,
    json={
        "question": "Tell me about Morocco's digital transformation",
        "top_k": 3,
        "include_sources": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

## JavaScript/TypeScript Client Example

```typescript
const API_URL = "http://localhost:8888";
const API_KEY = "your-secure-api-key-change-this-in-production";

async function queryRAG(question: string) {
  const response = await fetch(`${API_URL}/rag/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    },
    body: JSON.stringify({
      question: question,
      top_k: 3,
      include_sources: true
    })
  });
  
  const result = await response.json();
  return result;
}

// Usage
const answer = await queryRAG("What is Casablanca Finance City?");
console.log(answer.answer);
```

## Production Deployment

For production, you should:

1. **Change the API key** in `.env`:
   ```bash
   API_KEY=generate-a-strong-random-key-here
   ```

2. **Add SSL/TLS** using nginx or Traefik

3. **Configure CORS** properly in `src/api.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

4. **Add rate limiting** using libraries like `slowapi`

5. **Setup monitoring** with Prometheus/Grafana

6. **Use a production WSGI server** like gunicorn:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
   ```
