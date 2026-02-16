"""
Backend API with Job Queue System
Production-ready backend service for multiple client applications
"""
import os
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from embeddings import EmbeddingService
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import asyncio
from collections import defaultdict

load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY", "your-secure-api-key-here")
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

app = FastAPI(
    title="LLM Backend Service",
    description="Reusable backend service for embeddings, LLM inference, RAG, and async jobs",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService()
rag_systems = {}  # Collection name -> RAG instance
orchestrator = AgentOrchestrator()

# LLM model for chat/completion
from pydantic_ai.providers.openai import OpenAIProvider

# Initialize the model with custom provider
provider = OpenAIProvider(
    base_url=f"{LITELLM_URL}/v1",
    api_key=LITELLM_KEY,
)

model = OpenAIModel(
    'internal-llm',
    provider=provider
)

# Job queue system
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
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

# In-memory job storage (use Redis/DB in production)
jobs_db: Dict[str, Job] = {}
job_queue = asyncio.Queue()

# Request/Response Models
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)

class ChatResponse(BaseModel):
    response: str

class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to complete")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=4000)

class CompletionResponse(BaseModel):
    completion: str

class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    collection: str = Field("default", description="Collection name")
    top_k: int = Field(3, ge=1, le=10)

class RAGIngestRequest(BaseModel):
    documents: List[Dict[str, Any]] = Field(..., description="Documents to ingest")
    collection: str = Field("default", description="Collection name")

class JobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="Type of job")
    params: Dict[str, Any] = Field(..., description="Job parameters")
    priority: str = Field("normal", description="Job priority")

class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Helper functions
def get_rag_system(collection: str) -> ProductionRAG:
    """Get or create RAG system for collection"""
    if collection not in rag_systems:
        rag_systems[collection] = ProductionRAG(collection_name=collection)
    return rag_systems[collection]

# Job processor
async def process_job(job: Job):
    """Process a job asynchronously"""
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    
    try:
        if job.job_type == "batch_embed":
            # Batch embedding job
            texts = job.params.get("texts", [])
            embeddings = embedding_service.embed_batch(texts)
            job.result = {"embeddings": embeddings, "count": len(embeddings)}
            
        elif job.job_type == "batch_rag_ingest":
            # Batch RAG ingestion
            documents = job.params.get("documents", [])
            collection = job.params.get("collection", "default")
            rag = get_rag_system(collection)
            doc_ids = rag.ingest_documents(documents)
            job.result = {"document_ids": doc_ids, "count": len(doc_ids)}
            
        elif job.job_type == "analyze_document":
            # Document analysis
            document = job.params.get("document", "")
            result = await orchestrator.analyze_document(document)
            job.result = result.dict() if hasattr(result, 'dict') else result
            
        elif job.job_type == "batch_chat":
            # Batch chat completions
            messages = job.params.get("messages", [])
            agent = Agent(model, system_prompt="You are a helpful assistant.")
            responses = []
            for i, msg in enumerate(messages):
                result = await agent.run(msg)
                responses.append(result.data)
                job.progress = int((i + 1) / len(messages) * 100)
            job.result = {"responses": responses}
            
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
        
        job.status = JobStatus.COMPLETED
        job.progress = 100
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
    
    finally:
        job.completed_at = datetime.utcnow()

async def job_worker():
    """Background worker to process jobs"""
    while True:
        job = await job_queue.get()
        await process_job(job)
        job_queue.task_done()

# Start job worker on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(job_worker())

# Health & Info
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-backend-service"}

@app.get("/info")
async def system_info():
    return {
        "service": "LLM Backend Service",
        "version": "2.0.0",
        "features": [
            "Embeddings (BGE-M3)",
            "Chat/Completion (DeepSeek-R1)",
            "RAG with vector search",
            "Multi-agent orchestration",
            "Async job queue",
            "Multi-collection support"
        ],
        "endpoints": {
            "embeddings": ["/api/embeddings"],
            "chat": ["/api/chat", "/api/complete"],
            "rag": ["/api/rag/query", "/api/rag/ingest"],
            "jobs": ["/api/jobs"],
            "agents": ["/api/agent/analyze", "/api/agent/generate"]
        }
    }

# Embedding Endpoints
@app.post("/api/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(verify_api_key)])
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    try:
        embeddings = embedding_service.embed_batch(request.texts)
        return EmbeddingResponse(embeddings=embeddings, count=len(embeddings))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat/Completion Endpoints
@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_completion(request: ChatRequest):
    """Chat completion"""
    try:
        agent = Agent(
            model,
            system_prompt=request.system_prompt or "You are a helpful assistant."
        )
        result = await agent.run(request.message)
        return ChatResponse(response=result.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complete", response_model=CompletionResponse, dependencies=[Depends(verify_api_key)])
async def text_completion(request: CompletionRequest):
    """Text completion"""
    try:
        agent = Agent(model, system_prompt="Complete the following text:")
        result = await agent.run(request.prompt)
        return CompletionResponse(completion=result.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RAG Endpoints
@app.post("/api/rag/query", dependencies=[Depends(verify_api_key)])
async def rag_query(request: RAGQueryRequest):
    """Query RAG system"""
    try:
        rag = get_rag_system(request.collection)
        result = await rag.query(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/ingest", dependencies=[Depends(verify_api_key)])
async def rag_ingest(request: RAGIngestRequest):
    """Ingest documents into RAG system"""
    try:
        rag = get_rag_system(request.collection)
        doc_ids = rag.ingest_documents(request.documents)
        return {"document_ids": doc_ids, "count": len(doc_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job Endpoints
@app.post("/api/jobs", dependencies=[Depends(verify_api_key)])
async def submit_job(request: JobSubmitRequest):
    """Submit an async job"""
    job_id = str(uuid.uuid4())
    job = Job(job_id, request.job_type, request.params, request.priority)
    jobs_db[job_id] = job
    await job_queue.put(job)
    return {"job_id": job_id, "status": job.status}

@app.get("/api/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )

@app.delete("/api/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def cancel_job(job_id: str):
    """Cancel a job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        job.status = JobStatus.CANCELLED
        return {"message": "Job cancelled", "job_id": job_id}
    else:
        return {"message": "Job already completed or failed", "job_id": job_id}

# Agent Endpoints
@app.post("/api/agent/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_document(request: Dict):
    """Analyze a document"""
    try:
        result = await orchestrator.analyze_document(request["document"])
        return result.dict() if hasattr(result, 'dict') else result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/generate", dependencies=[Depends(verify_api_key)])
async def generate_content(request: Dict):
    """Generate content"""
    try:
        result = await orchestrator.generate_content(
            request["prompt"],
            request.get("context", "")
        )
        return {"content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
