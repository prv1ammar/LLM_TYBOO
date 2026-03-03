"""
unified_api.py — Unified Tython AI API
=======================================
Combines Dashboard API (JWT) and Backend API (API Key).
Supports browser clients (dashboard) and programmatic access (SDK/n8n).

AUTHENTICATION:
  1. JWT: Header "Authorization: Bearer <token>"
  2. API Key: Header "X-API-Key: <key>"

All endpoints under /api/* and /rag/* now support both.
"""

import os
import uuid
import asyncio
import traceback
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
API_KEY = os.getenv("API_KEY", "your-secure-api-key-here")

app = FastAPI(
    title="LLM_TYBOO Unified API",
    description="Unified AI Platform — Dashboard + SDK Entry Point",
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
rag_systems: Dict[str, ProductionRAG] = {
    "enterprise_kb": ProductionRAG(collection_name="enterprise_kb")
}
orchestrator = AgentOrchestrator()
embedding_service = EmbeddingService()

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

# ── Request / Response Models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(...)
    top_k: int = Field(3, ge=1, le=10)
    include_sources: bool = Field(True)

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    document_ids: List[str]
    count: int

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000

class ChatResponse(BaseModel):
    response: str

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

# ... Admin models ...
class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field("user")

class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)

class SetActiveRequest(BaseModel):
    is_active: bool

# ── Authentication Helper ─────────────────────────────────────────────────────
async def verify_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    current_jwt_user: Optional[dict] = Depends(lambda: None) # Placeholder for get_current_user fallback
):
    """
    Checks for EITHER a valid API Key OR a valid JWT.
    Allows n8n/SDK and Dashboard to use the same endpoints.
    """
    # 1. Check API Key
    if x_api_key:
        if x_api_key == API_KEY:
            return {"type": "api_key", "identity": "m2m"}
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # 2. Check JWT (Manually call get_current_user to avoid early failure)
    # This is a bit tricky with Depends, so we'll just use a more explicit check
    return {"type": "jwt", "identity": "user"} # Simplified for now, real logic below

async def get_any_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    if x_api_key:
        if x_api_key == API_KEY:
            return {"username": "m2m_client", "role": "m2m"}
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    if authorization:
        user = await get_current_user(authorization.replace("Bearer ", ""))
        return user
    
    raise HTTPException(status_code=401, detail="Not authenticated (X-API-Key or JWT required)")

async def require_admin(auth: dict = Depends(get_any_auth)):
    if auth.get("role") == "admin":
        return auth
    raise HTTPException(status_code=403, detail="Admin role required")

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        init_users_table()
        asyncio.create_task(job_worker())
    except Exception as e:
        print(f"[WARN] Startup failure: {e}")

# ── Job Worker ────────────────────────────────────────────────────────────────
async def process_job(job: Job):
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    try:
        if job.job_type == "batch_embed":
            texts = job.params.get("texts", [])
            embeddings = embedding_service.embed_batch(texts)
            job.result = {"embeddings": embeddings, "count": len(embeddings)}
        elif job.job_type == "batch_rag_ingest":
            documents = job.params.get("documents", [])
            collection = job.params.get("collection", "enterprise_kb")
            if collection not in rag_systems: rag_systems[collection] = ProductionRAG(collection_name=collection)
            ids = rag_systems[collection].ingest_documents(documents)
            job.result = {"document_ids": ids, "count": len(ids)}
        elif job.job_type == "analyze_document":
            res = await orchestrator.analyze_document(job.params.get("document"), job.params.get("instructions"))
            job.result = res.model_dump() if hasattr(res, "model_dump") else res
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    finally:
        job.completed_at = datetime.utcnow()

async def job_worker():
    while True:
        job = await job_queue.get()
        await process_job(job)
        job_queue.task_done()

# ── Endpoints: Auth ───────────────────────────────────────────────────────────
@app.post("/token", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(data={"sub": user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

# ── Endpoints: System ─────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "service": "llm-tyboo-unified-api"}

@app.get("/info", tags=["System"])
async def info():
    return {
        "service": "LLM_TYBOO — Unified Edition", "version": "2.1.0",
        "auth_modes": ["JWT", "API-Key"],
        "endpoints": ["/api/chat", "/api/embeddings", "/api/rag/query", "/api/jobs", "/admin/users"]
    }

# ── Endpoints: RAG & AI (Programmatic / SDK compatible) ───────────────────────
@app.post("/api/chat", dependencies=[Depends(get_any_auth)], tags=["AI"])
async def api_chat(request: ChatRequest):
    try:
        model, _ = get_model_for_chat(request.message)
        agent = Agent(model, system_prompt=request.system_prompt or "You are a helpful assistant.")
        result = await agent.run(request.message)
        return {"response": result.output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings", dependencies=[Depends(get_any_auth)], tags=["AI"])
async def api_embeddings(request: EmbeddingRequest):
    try:
        embs = embedding_service.embed_batch(request.texts)
        return {"embeddings": embs, "count": len(embs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/query", dependencies=[Depends(get_any_auth)], tags=["RAG"])
@app.post("/rag/query", dependencies=[Depends(get_any_auth)], tags=["RAG"])
async def api_rag_query(request: QueryRequest, collection: str = "enterprise_kb"):
    # Unified both paths /api/rag/query (SDK) and /rag/query (Dashboard)
    try:
        if collection not in rag_systems: rag_systems[collection] = ProductionRAG(collection_name=collection)
        result = await rag_systems[collection].query(request.question, top_k=request.top_k, include_sources=request.include_sources)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/ingest", dependencies=[Depends(get_any_auth)], tags=["RAG"])
@app.post("/rag/ingest", dependencies=[Depends(get_any_auth)], tags=["RAG"])
async def api_rag_ingest(documents: List[DocumentRequest], collection: str = "enterprise_kb"):
    try:
        if collection not in rag_systems: rag_systems[collection] = ProductionRAG(collection_name=collection)
        docs = [{"text": d.text, "metadata": d.metadata} for d in documents]
        ids = rag_systems[collection].ingest_documents(docs)
        return {"document_ids": ids, "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Endpoints: Jobs ───────────────────────────────────────────────────────────
@app.post("/api/jobs", dependencies=[Depends(get_any_auth)], tags=["Jobs"])
async def submit_job_endpoint(request: JobSubmitRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id, request.job_type, request.params, request.priority)
    jobs_db[job_id] = job
    await job_queue.put(job)
    return {"job_id": job_id, "status": job.status}

@app.get("/api/jobs/{job_id}", dependencies=[Depends(get_any_auth)], tags=["Jobs"])
async def get_job_endpoint(job_id: str):
    if job_id not in jobs_db: raise HTTPException(status_code=404, detail="Job not found")
    j = jobs_db[job_id]
    return {
        "job_id": j.job_id, "status": j.status, "progress": j.progress,
        "result": j.result, "error": j.error, "created_at": j.created_at
    }

# ── Endpoints: Admin ──────────────────────────────────────────────────────────
@app.get("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin"])
async def admin_list_users():
    return {"users": list_users()}

@app.post("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin"], status_code=201)
async def admin_create_user_endpoint(body: CreateUserRequest):
    try:
        user = create_user(body.username, body.password, body.role)
        return {"message": f"User '{body.username}' created", "user": user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/admin/users/{username}", dependencies=[Depends(require_admin)], tags=["Admin"])
async def admin_delete_user_endpoint(username: str, current_user: dict = Depends(require_admin)):
    if username == current_user["username"]: raise HTTPException(status_code=400, detail="Cannot delete self")
    if delete_user(username): return {"message": f"User '{username}' deleted"}
    raise HTTPException(status_code=404, detail="User not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
