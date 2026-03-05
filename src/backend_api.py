"""
backend_api.py — Extended Backend API with Async Job Queue
===========================================================
PURPOSE:
  An extended version of api.py that adds:
    1. Embedding endpoint — expose BGE-M3 embeddings via HTTP
    2. Async job queue — submit long-running tasks and poll for results
    3. Multi-collection RAG — each collection is a separate knowledge base
    4. Agent endpoints — document analysis and content generation
    5. API Key authentication (X-API-Key header) instead of JWT

  This API is designed for machine-to-machine usage (n8n, LangChain agents,
  SDK clients) rather than browser/user interaction.

WHEN TO USE THIS vs api.py:
  api.py          → Browser clients, dashboard, Swagger UI, JWT auth
  backend_api.py  → SDK clients, n8n, LangChain, programmatic access, API key auth

AUTHENTICATION:
  This API uses X-API-Key header authentication instead of JWT.
  Set API_KEY in .env — the same key goes in the client's X-API-Key header.

  Example:
    curl -H "X-API-Key: your-api-key" http://localhost:8888/api/embeddings \
         -d '{"texts": ["hello world"]}'

ASYNC JOB QUEUE:
  Some operations (batch embedding, large ingestion) take a long time.
  Instead of blocking the HTTP connection, you can:
    1. Submit a job → POST /api/jobs  → returns job_id immediately
    2. Poll status  → GET  /api/jobs/{job_id} → returns status + result when done

  Job states: pending → running → completed / failed / cancelled

SUPPORTED JOB TYPES:
  "batch_embed"       → embed a list of texts, returns embedding vectors
  "batch_rag_ingest"  → ingest documents into a collection
  "analyze_document"  → run document analysis, returns AnalysisResult
  "batch_chat"        → run multiple chat prompts in sequence

HOW TO START:
  # Default port 8888 (same as api.py — run one or the other, not both)
  python backend_api.py

  # Or via uvicorn
  uvicorn backend_api:app --host 0.0.0.0 --port 8889 --workers 2
"""

import os
import uuid
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from embeddings import EmbeddingService
from model_router import get_model

load_dotenv()

# API key for machine-to-machine authentication
# Set in .env — same key used in SDK and n8n HTTP credentials
API_KEY = os.getenv("API_KEY", "your-secure-api-key-here")

app = FastAPI(
    title="LLM_TYBOO Backend API",
    description="Machine-to-machine API with embeddings, async jobs, and multi-collection RAG",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service instances ─────────────────────────────────────────────────────────
# Initialized once at startup, shared across all requests
embedding_service = EmbeddingService()
orchestrator = AgentOrchestrator()

# Multiple RAG collections — created on demand when first used
# Key = collection name, Value = ProductionRAG instance
rag_systems: Dict[str, ProductionRAG] = {}


# ── Job queue system ──────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """
    Possible states of an async job:
      PENDING   → created, waiting in queue
      RUNNING   → currently being processed
      COMPLETED → finished successfully, result is available
      FAILED    → terminated with an error, error message is available
      CANCELLED → cancelled by the client before completion
    """
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class Job:
    """
    Represents a single async job with its full lifecycle state.
    Stored in memory (jobs_db dict) — use Redis in production for persistence.
    """
    def __init__(self, job_id: str, job_type: str, params: Dict, priority: str):
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.priority = priority
        self.status = JobStatus.PENDING
        self.progress = 0           # 0-100 progress percentage
        self.result = None          # Set when status = COMPLETED
        self.error = None           # Set when status = FAILED
        self.created_at = datetime.utcnow()
        self.started_at = None      # Set when processing begins
        self.completed_at = None    # Set when processing ends (success or failure)


# In-memory job storage — replace with Redis for production multi-instance deployments
jobs_db: Dict[str, Job] = {}

# Asyncio queue — jobs are put here and processed by the background worker
job_queue: asyncio.Queue = asyncio.Queue()


# ── Request / Response models ─────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed with BGE-M3")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]  # One 1024D vector per input text
    count: int

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: Optional[str] = Field(None, description="Override default system prompt")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)

class ChatResponse(BaseModel):
    response: str

class CompletionRequest(BaseModel):
    prompt: str
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)

class CompletionResponse(BaseModel):
    completion: str

class RAGQueryRequest(BaseModel):
    question: str
    collection: str = Field("default", description="Which knowledge base collection to search")
    top_k: int = Field(3, ge=1, le=10)

class RAGIngestRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description='List of {"text": ..., "metadata": ...} dicts')
    collection: str = Field("default", description="Target collection name")

class JobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="One of: batch_embed, batch_rag_ingest, analyze_document, batch_chat")
    params: Dict[str, Any] = Field(..., description="Job-type-specific parameters")
    priority: str = Field("normal", description="Job priority: low, normal, high")

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ── Authentication ────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(..., description="API key from .env → API_KEY")):
    """
    Verify the X-API-Key header against the configured API_KEY.
    Used as a FastAPI dependency on all protected endpoints.

    Usage in a route:
      @app.post("/endpoint", dependencies=[Depends(verify_api_key)])
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key — check X-API-Key header matches API_KEY in .env"
        )
    return x_api_key


# ── Helper: get or create RAG system for a collection ────────────────────────

def get_rag(collection: str) -> ProductionRAG:
    """
    Return the RAG instance for the given collection name.
    Creates a new one if it doesn't exist yet.

    This allows n8n or other clients to work with multiple separate
    knowledge bases without any server-side configuration changes.
    Just pass a different collection name in the request.
    """
    if collection not in rag_systems:
        rag_systems[collection] = ProductionRAG(collection_name=collection)
    return rag_systems[collection]


# ── Job processor ─────────────────────────────────────────────────────────────

async def process_job(job: Job):
    """
    Execute a job based on its type.
    Called by the background worker — never called directly by HTTP handlers.

    Job types and their required params:
      batch_embed:
        params: {"texts": ["text1", "text2", ...]}
        result: {"embeddings": [[...], [...]], "count": N}

      batch_rag_ingest:
        params: {"documents": [...], "collection": "name"}
        result: {"document_ids": [...], "count": N}

      analyze_document:
        params: {"document": "full text here"}
        result: AnalysisResult dict

      batch_chat:
        params: {"messages": ["prompt1", "prompt2", ...]}
        result: {"responses": ["response1", "response2", ...]}
    """
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()

    try:
        if job.job_type == "batch_embed":
            texts = job.params.get("texts", [])
            embeddings = embedding_service.embed_batch(texts)
            job.result = {"embeddings": embeddings, "count": len(embeddings)}

        elif job.job_type == "batch_rag_ingest":
            documents = job.params.get("documents", [])
            collection = job.params.get("collection", "default")
            rag = get_rag(collection)
            doc_ids = rag.ingest_documents(documents)
            job.result = {"document_ids": doc_ids, "count": len(doc_ids)}

        elif job.job_type == "analyze_document":
            document = job.params.get("document", "")
            instructions = job.params.get("instructions", None)
            result = await orchestrator.analyze_document(document, instructions)
            # Convert Pydantic model to dict for JSON serialization
            job.result = result.model_dump() if hasattr(result, "model_dump") else result

        elif job.job_type == "batch_chat":
            messages = job.params.get("messages", [])
            model = get_model()  # 14B for batch chat quality
            agent = Agent(model, system_prompt="You are a helpful assistant.")
            responses = []
            for i, msg in enumerate(messages):
                result = await agent.run(msg)
                responses.append(result.output)
                # Update progress as each message completes
                job.progress = int((i + 1) / len(messages) * 100)
            job.result = {"responses": responses}

        else:
            raise ValueError(f"Unknown job type: '{job.job_type}'")

        job.status = JobStatus.COMPLETED
        job.progress = 100

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)

    finally:
        job.completed_at = datetime.utcnow()


async def job_worker():
    """
    Background coroutine that processes jobs from the queue one at a time.
    Started automatically when the FastAPI app starts (see startup event).
    Runs indefinitely — one job at a time, in submission order.
    """
    while True:
        job = await job_queue.get()
        await process_job(job)
        job_queue.task_done()


@app.on_event("startup")
async def startup_event():
    """
    Start the background job worker when FastAPI starts.
    asyncio.create_task() runs it concurrently alongside the HTTP server.
    """
    asyncio.create_task(job_worker())


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "llm-tyboo-backend-api"}


@app.get("/info")
async def info():
    return {
        "service": "LLM_TYBOO Backend API",
        "version": "2.0.0",
        "auth": "X-API-Key header",
        "job_types": ["batch_embed", "batch_rag_ingest", "analyze_document", "batch_chat"],
        "endpoints": {
            "embeddings": "/api/embeddings",
            "chat": ["/api/chat", "/api/complete"],
            "rag": ["/api/rag/query", "/api/rag/ingest"],
            "jobs": "/api/jobs",
            "agents": ["/api/agent/analyze", "/api/agent/generate"],
        }
    }


@app.post(
    "/api/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Embeddings"],
    summary="Generate BGE-M3 embeddings"
)
async def create_embeddings(request: EmbeddingRequest):
    """
    Embed a list of texts using BGE-M3 (1024 dimensions).
    Useful when you need vectors for external systems or custom similarity search.

    Returns one 1024-float vector per input text, in the same order.
    """
    try:
        embeddings = embedding_service.embed_batch(request.texts)
        return EmbeddingResponse(embeddings=embeddings, count=len(embeddings))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["LLM"],
    summary="Single chat completion"
)
async def chat(request: ChatRequest):
    """
    Generate a chat response using the auto-routed model (3B or 14B).
    Optionally provide a custom system prompt to control the model's behavior.
    """
    try:
        from model_router import get_model_for_chat
        model, _ = get_model_for_chat(request.message)
        agent = Agent(model, system_prompt=request.system_prompt or "You are a helpful assistant.")
        result = await agent.run(request.message)
        return ChatResponse(response=result.output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/complete",
    response_model=CompletionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["LLM"],
    summary="Text completion"
)
async def complete(request: CompletionRequest):
    """Complete a prompt using the 14B model."""
    try:
        model = get_model()
        agent = Agent(model, system_prompt="Complete the following text:")
        result = await agent.run(request.prompt)
        return CompletionResponse(completion=result.output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/query",
    dependencies=[Depends(verify_api_key)],
    tags=["RAG"],
    summary="Query a RAG collection"
)
async def rag_query(request: RAGQueryRequest):
    """Search a knowledge base collection and generate an answer."""
    try:
        rag = get_rag(request.collection)
        return await rag.query(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/ingest",
    dependencies=[Depends(verify_api_key)],
    tags=["RAG"],
    summary="Ingest documents into a collection"
)
async def rag_ingest(request: RAGIngestRequest):
    """Add documents to a knowledge base collection."""
    try:
        rag = get_rag(request.collection)
        ids = rag.ingest_documents(request.documents)
        return {"document_ids": ids, "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/jobs",
    dependencies=[Depends(verify_api_key)],
    tags=["Jobs"],
    summary="Submit an async job"
)
async def submit_job(request: JobSubmitRequest):
    """
    Submit a long-running job and get a job_id back immediately.
    Poll GET /api/jobs/{job_id} to check status and retrieve the result.
    """
    job_id = str(uuid.uuid4())
    job = Job(job_id, request.job_type, request.params, request.priority)
    jobs_db[job_id] = job
    await job_queue.put(job)
    return {"job_id": job_id, "status": job.status}


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Jobs"],
    summary="Get job status and result"
)
async def get_job(job_id: str):
    """
    Poll for job status. When status == "completed", result contains the output.
    When status == "failed", error contains the error message.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_db[job_id]
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.delete(
    "/api/jobs/{job_id}",
    dependencies=[Depends(verify_api_key)],
    tags=["Jobs"],
    summary="Cancel a pending or running job"
)
async def cancel_job(job_id: str):
    """
    Mark a job as cancelled.
    Note: If the job is already running, it may still complete before
    the cancellation takes effect (no forceful interrupt).
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_db[job_id]
    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        job.status = JobStatus.CANCELLED
        return {"message": "Job cancelled", "job_id": job_id}
    return {"message": f"Job already {job.status}", "job_id": job_id}


@app.post(
    "/api/agent/analyze",
    dependencies=[Depends(verify_api_key)],
    tags=["Agent"],
    summary="Analyze a document"
)
async def analyze_document(body: Dict):
    """
    Deep document analysis — returns summary, key points, sentiment, and extracted data.
    Send: {"document": "full text here", "instructions": "optional focus area"}
    """
    try:
        result = await orchestrator.analyze_document(
            body["document"],
            body.get("instructions")
        )
        return result.model_dump() if hasattr(result, "model_dump") else result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/agent/generate",
    dependencies=[Depends(verify_api_key)],
    tags=["Agent"],
    summary="Generate content"
)
async def generate_content(body: Dict):
    """
    Generate any type of content given a prompt and optional context.
    Send: {"prompt": "...", "context": "optional background info"}
    """
    try:
        result = await orchestrator.generate_content(
            body["prompt"],
            body.get("context", "")
        )
        return {"content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
