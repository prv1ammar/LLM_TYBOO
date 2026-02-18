"""
FastAPI REST API for the self-hosted AI stack
Production-ready API with authentication, rate limiting, and monitoring
"""
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from dotenv import load_dotenv
import asyncio

load_dotenv()

app = FastAPI(
    title="Self-Hosted AI API",
    description="Production-ready API for Moroccan enterprise AI workflows",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_system = ProductionRAG(collection_name="enterprise_kb")
orchestrator = AgentOrchestrator()

# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(3, description="Number of relevant documents to retrieve", ge=1, le=10)
    include_sources: bool = Field(True, description="Whether to include source documents")

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None

class DocumentRequest(BaseModel):
    text: str = Field(..., description="Document text")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Document metadata")

class IngestResponse(BaseModel):
    document_ids: List[str]
    count: int

from fastapi.security import OAuth2PasswordRequestForm
from auth import create_access_token, get_current_user, verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta

# Mock Database for demo (In Phase 3.1 real Postgres would be used)
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("password123"),
    }
}

@app.post("/token", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Verification dependency
async def protect_endpoint(current_user: dict = Depends(get_current_user)):
    return current_user

# Health check
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "service": "self-hosted-ai-api"}

# RAG Endpoints
@app.post("/rag/query", response_model=QueryResponse, dependencies=[Depends(protect_endpoint)], tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question
    
    Returns an answer based on the knowledge base
    """
    try:
        result = await rag_system.query(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(protect_endpoint)], tags=["RAG"])
async def ingest_documents(documents: List[DocumentRequest]):
    """
    Ingest documents into the knowledge base
    
    Accepts a list of documents with text and optional metadata
    """
    try:
        docs = [{"text": doc.text, "metadata": doc.metadata} for doc in documents]
        doc_ids = rag_system.ingest_documents(docs)
        return IngestResponse(document_ids=doc_ids, count=len(doc_ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# Agent Endpoints
@app.post("/agent/general", dependencies=[Depends(protect_endpoint)], tags=["Agents"])
async def general_agent(request: QueryRequest):
    """
    Query the General Purpose AGATE Assistant
    Handles all topics: Legal, HR, IT, General Knowledge
    """
    try:
        result = await orchestrator.run_agent(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

# System Info
@app.get("/info")
async def system_info():
    """Get system information"""
    return {
        "service": "Self-Hosted AI Stack",
        "version": "1.1.0",
        "features": [
            "RAG with vector search (Qdrant)",
            "General Purpose AGATE Assistant (Unified)",
            "Self-hosted LLM (Qwen2.5-7B)",
            "Self-hosted embeddings (BGE-M3)",
            "Observability (Prometheus/Grafana)",
            "Usage Tracking (PostgreSQL)"
        ],
        "endpoints": {
            "rag": ["/rag/query", "/rag/ingest"],
            "agents": ["/agent/general"],
            "system": ["/health", "/info", "/docs"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
