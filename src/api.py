"""
api.py — Unified FastAPI REST API (JWT + API Key)
===================================================
A single entry point that handles:
1. User-facing routes (JWT Auth):
   - /token, /rag/query, /rag/ingest, /agent/general
2. Machine-to-machine routes (API Key Auth):
   - /api/embeddings, /api/chat, /api/complete
   - /api/rag/query, /api/rag/ingest
   - /api/jobs (Async job queue), /api/agent/*
3. Admin endpoints: /admin/users/*

AUTHENTICATION:
- JWT: Used by dashboard/browsers. Obtain via /token.
- API Key: Used by SDK/n8n. Pass in X-API-Key header.
"""

import os
import uuid
import asyncio
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, status, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from embeddings import EmbeddingService
from model_router import get_model, get_model_for_chat
from auth import (
    create_access_token,
    get_current_user,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from users import (
    init_users_table,
    get_user,
    list_users,
    create_user,
    delete_user,
    update_password,
    set_user_active,
)

load_dotenv()

# Config
API_KEY = os.getenv("API_KEY", "92129f24-12f0-478a-8c98-5b15070235e6")

app = FastAPI(
    title="LLM_TYBOO Unified API",
    description="Self-Hosted AI Platform — CPU Edition (Qwen2.5 14B + 3B)",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service Instances ─────────────────────────────────────────────────────────
rag_system = ProductionRAG(collection_name="enterprise_kb")
orchestrator = AgentOrchestrator()
embedding_service = EmbeddingService()
rag_systems: Dict[str, ProductionRAG] = {}

# ── Job Queue System ──────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"

class Job:
    def __init__(self, job_id: str, job_type: str, params: Dict, priority: str):
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.priority = priority
        self.status = JobStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None

jobs_db: Dict[str, Job] = {}
job_queue: asyncio.Queue = asyncio.Queue()

async def process_job(job: Job):
    print(f"[JOB] Starting {job.job_type} ({job.job_id})")
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
            rag = get_rag_backend(collection)
            doc_ids = rag.ingest_documents(documents)
            job.result = {"document_ids": doc_ids, "count": len(doc_ids)}
        elif job.job_type == "analyze_document":
            document = job.params.get("document", "")
            instructions = job.params.get("instructions", None)
            result = await orchestrator.analyze_document(document, instructions)
            job.result = result.model_dump() if hasattr(result, "model_dump") else result
        elif job.job_type == "batch_chat":
            messages = job.params.get("messages", [])
            model = get_model()
            agent = Agent(model, system_prompt="You are a helpful assistant.")
            responses = []
            for i, msg in enumerate(messages):
                res = await agent.run(msg)
                responses.append(res.output)
                job.progress = int((i + 1) / len(messages) * 100)
            job.result = {"responses": responses}
        else:
            raise ValueError(f"Unknown job type: '{job.job_type}'")
        job.status = JobStatus.COMPLETED
        job.progress = 100
        print(f"[JOB] Completed {job.job_id}")
    except Exception as e:
        import traceback
        print(f"[JOB] Failed {job.job_id}: {str(e)}")
        traceback.print_exc()
        job.status = JobStatus.FAILED
        job.error = str(e)
    finally:
        job.completed_at = datetime.utcnow()

async def job_worker():
    while True:
        job = await job_queue.get()
        await process_job(job)
        job_queue.task_done()

# ── Startup/Shutdown ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        init_users_table()
    except Exception as e:
        print(f"[WARN] User table init failed: {e}")
    asyncio.create_task(job_worker())

# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    document_ids: List[str]
    count: int

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class JobSubmitRequest(BaseModel):
    job_type: str
    params: Dict[str, Any]
    priority: str = "normal"

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# ── Auth Dependencies ─────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

async def require_auth(current_user: dict = Depends(get_current_user)):
    return current_user

async def require_admin(current_user: dict = Depends(get_current_user)):
    user = get_user(current_user["username"])
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user

def get_rag_backend(collection: str) -> ProductionRAG:
    if collection == "enterprise_kb": return rag_system
    if collection not in rag_systems:
        rag_systems[collection] = ProductionRAG(collection_name=collection)
    return rag_systems[collection]

# ── Public / JWT Routes ───────────────────────────────────────────────────────
@app.post("/token", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        print(f"[AUTH] Login attempt for user: {form_data.username}")
        user = get_user(form_data.username)
        if not user:
            print(f"[AUTH] User not found: {form_data.username}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        
        if not verify_password(form_data.password, user["hashed_password"]):
            print(f"[AUTH] Invalid password for: {form_data.username}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        
        token = create_access_token(data={"sub": user["username"]})
        print(f"[AUTH] Login successful for: {form_data.username}")
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Login failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Auth Error: {str(e)}")

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "service": "llm-tyboo-api"}

@app.get("/info", tags=["System"])
async def info():
    return {
        "service": "LLM_TYBOO Unified API",
        "version": "2.1.0",
        "models": {"llm_14b": "Qwen2.5-14B", "llm_3b": "Qwen2.5-3B", "embeddings": "BGE-M3"}
    }

@app.post("/rag/query", response_model=QueryResponse, dependencies=[Depends(require_auth)], tags=["RAG"])
async def query_rag(request: QueryRequest):
    return await rag_system.query(request.question, top_k=request.top_k, include_sources=request.include_sources)

@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(require_auth)], tags=["RAG"])
async def ingest_documents(documents: List[DocumentRequest]):
    docs = [{"text": d.text, "metadata": d.metadata} for d in documents]
    ids = rag_system.ingest_documents(docs)
    return {"document_ids": ids, "count": len(ids)}

@app.post("/agent/general", dependencies=[Depends(require_auth)], tags=["Agent"])
async def agent_general(request: QueryRequest):
    res = await orchestrator.run_agent(request.question)
    return {"answer": res}

# ── API Key Routes (/api/*) ───────────────────────────────────────────────────
@app.post("/api/embeddings", dependencies=[Depends(verify_api_key)], tags=["Machine"])
async def api_embeddings(request: Dict):
    texts = request.get("texts", [])
    embs = embedding_service.embed_batch(texts)
    return {"embeddings": embs, "count": len(embs)}

@app.post("/v1/embeddings", tags=["OpenAI"])
async def openai_embeddings(request: Dict):
    """OpenAI-compatible embedding endpoint for BGE-M3"""
    inputs = request.get("input", [])
    if isinstance(inputs, str): inputs = [inputs]
    embs = embedding_service.embed_batch(inputs)
    data = []
    for i, vec in enumerate(embs):
        data.append({"object": "embedding", "index": i, "embedding": vec})
    return {
        "object": "list",
        "data": data,
        "model": request.get("model", "bge-m3"),
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

@app.post("/api/chat", dependencies=[Depends(verify_api_key)], tags=["Machine"])
async def api_chat(request: Dict):
    try:
        msg = request.get("message", "")
        model, label = get_model_for_chat(msg)
        print(f"[API] Chat routing to {label}")
        agent = Agent(model, system_prompt=request.get("system_prompt", "You are a helpful assistant."))
        res = await agent.run(msg)
        return {"response": res.output}
    except Exception as e:
        print(f"[ERROR] API Chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complete", dependencies=[Depends(verify_api_key)], tags=["Machine"])
async def api_complete(request: Dict):
    model = get_model()
    agent = Agent(model, system_prompt="Complete the following text:")
    res = await agent.run(request.get("prompt", ""))
    return {"completion": res.output}

@app.post("/api/rag/query", dependencies=[Depends(verify_api_key)], tags=["Machine"])
async def api_rag_query(request: Dict):
    rag = get_rag_backend(request.get("collection", "default"))
    return await rag.query(request.get("question", ""), top_k=request.get("top_k", 3))

@app.post("/api/rag/ingest", dependencies=[Depends(verify_api_key)], tags=["Machine"])
async def api_rag_ingest(request: Dict):
    rag = get_rag_backend(request.get("collection", "default"))
    ids = rag.ingest_documents(request.get("documents", []))
    return {"document_ids": ids, "count": len(ids)}

@app.post("/api/jobs", dependencies=[Depends(verify_api_key)], tags=["Jobs"])
async def api_submit_job(request: JobSubmitRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id, request.job_type, request.params, request.priority)
    jobs_db[job_id] = job
    await job_queue.put(job)
    return {"job_id": job_id, "status": job.status}

@app.get("/api/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(verify_api_key)], tags=["Jobs"])
async def api_get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found in this worker instance")
    j = jobs_db[job_id]
    return JobResponse(job_id=j.job_id, status=j.status, progress=j.progress, result=j.result, error=j.error,
                       created_at=j.created_at, started_at=j.started_at, completed_at=j.completed_at)

@app.delete("/api/jobs/{job_id}", dependencies=[Depends(verify_api_key)], tags=["Jobs"])
async def api_delete_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs_db[job_id]
    if j.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        j.status = JobStatus.CANCELLED
        return {"message": "Job cancelled", "job_id": job_id}
    return {"message": f"Job already {j.status}", "job_id": job_id}

@app.post("/api/agent/analyze", dependencies=[Depends(verify_api_key)], tags=["Agent"])
async def api_analyze_doc(body: Dict):
    try:
        print(f"[API] Analyzing document ({len(body.get('document',''))} chars)")
        res = await orchestrator.analyze_document(body["document"], body.get("instructions"))
        return res.model_dump() if hasattr(res, "model_dump") else res
    except Exception as e:
        import traceback
        print(f"[ERROR] Document Analysis: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@app.post("/api/agent/generate", dependencies=[Depends(verify_api_key)], tags=["Agent"])
async def api_generate(body: Dict):
    res = await orchestrator.generate_content(body["prompt"], body.get("context", ""))
    return {"content": res}

# ── Admin Routes ──────────────────────────────────────────────────────────────
@app.get("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin"])
async def list_all_users():
    return {"users": list_users()}

@app.post("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin"])
async def add_user(body: CreateUserRequest):
    try:
        user = create_user(body.username, body.password, body.role)
        return {"message": "User created", "user": user}
    except ValueError as e: raise HTTPException(409, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
