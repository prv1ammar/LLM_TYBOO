"""
api.py — FastAPI REST API
===========================
PURPOSE:
  The main HTTP entry point for the LLM_TYBOO platform.
  Exposes RAG queries and agent calls as REST endpoints
  protected by JWT authentication.

ENDPOINTS:
  POST /token              — Login, returns JWT access token
  GET  /health             — Service health check (no auth required)
  GET  /info               — Platform info (no auth required)
  POST /rag/query          — Query the knowledge base (JWT required)
  POST /rag/ingest         — Add documents to the knowledge base (JWT required)
  POST /agent/general      — Run the general automation agent (JWT required)

AUTHENTICATION FLOW:
  1. Client POSTs to /token with form data: username + password
  2. Server returns: {"access_token": "...", "token_type": "bearer"}
  3. Client includes the token in every subsequent request:
     Header: Authorization: Bearer <token>
  4. Protected endpoints automatically verify the token

DEFAULT CREDENTIALS (CHANGE BEFORE PRODUCTION):
  Username: admin
  Password: password123

HOW TO START THE API:
  # Development (with auto-reload)
  uvicorn api:app --host 0.0.0.0 --port 8888 --reload

  # Production (2 workers)
  uvicorn api:app --host 0.0.0.0 --port 8888 --workers 2

API DOCUMENTATION:
  Once running, visit: http://YOUR_SERVER:8888/docs
  The Swagger UI lets you test all endpoints interactively.
  You can log in, copy the token, and test protected routes directly.

EXAMPLE USAGE WITH CURL:
  # Step 1: Get token
  curl -X POST http://localhost:8888/token \
       -d "username=admin&password=password123"

  # Step 2: Query RAG (replace TOKEN with the access_token from step 1)
  curl -X POST http://localhost:8888/rag/query \
       -H "Authorization: Bearer TOKEN" \
       -H "Content-Type: application/json" \
       -d '{"question": "What is the refund policy?", "top_k": 3}'
"""

import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from datetime import timedelta
from dotenv import load_dotenv

from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from auth import (
    create_access_token,
    get_current_user,
    verify_password,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

load_dotenv()

# ── FastAPI app setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM_TYBOO API",
    description="Self-Hosted AI Platform — CPU Edition (Qwen2.5 14B + 3B)",
    version="1.1.0",
    # Swagger UI will be available at /docs
    # ReDoc will be available at /redoc
)

# CORS — controls which origins can call this API from a browser
# In production: replace "*" with your actual frontend domain(s)
# Example: allow_origins=["https://app.mycompany.ma", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service initialization ────────────────────────────────────────────────────
# These are created once at startup and reused across requests
# ProductionRAG loads the vector store connection
# AgentOrchestrator loads the RAG system internally
rag_system = ProductionRAG(collection_name="enterprise_kb")
orchestrator = AgentOrchestrator()

# ── User database ─────────────────────────────────────────────────────────────
# In production: replace this dict with a real PostgreSQL users table
# To add a user: {"username": "x", "hashed_password": get_password_hash("password")}
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("password123"),  # CHANGE THIS IN PRODUCTION
    }
}


# ── Request / Response models ─────────────────────────────────────────────────
# These Pydantic models define what the API accepts and returns.
# FastAPI uses them for automatic validation and Swagger documentation.

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the knowledge base or agent")
    top_k: int = Field(3, ge=1, le=10, description="Number of document chunks to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None

class DocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to ingest")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Source metadata (filename, page, etc.)")

class IngestResponse(BaseModel):
    document_ids: List[str]  # UUIDs assigned to each stored chunk
    count: int               # Number of documents successfully stored


# ── Auth endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/token",
    tags=["Auth"],
    summary="Login and get JWT token",
    description="Send username and password as form data. Returns a Bearer token valid for 60 minutes."
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with username + password, receive a JWT token.

    The token must be included in the Authorization header of all
    protected requests as: Authorization: Bearer <token>

    Uses form-encoded body (not JSON) — this is the OAuth2 standard.
    """
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


# ── Dependency: protect any endpoint by requiring a valid JWT ─────────────────
async def require_auth(current_user: dict = Depends(get_current_user)):
    """
    Reusable FastAPI dependency for JWT-protected endpoints.
    Add to any route with: dependencies=[Depends(require_auth)]
    Or as a parameter:     current_user: dict = Depends(require_auth)
    """
    return current_user


# ── Health / Info endpoints ───────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
async def health():
    """
    Returns {"status": "healthy"} when the API is running.
    Used by Docker health checks and monitoring tools.
    No authentication required.
    """
    return {"status": "healthy", "service": "llm-tyboo-api"}


@app.get("/info", tags=["System"], summary="Platform information")
async def info():
    """
    Returns general information about the platform.
    Useful for clients to discover available models and endpoints.
    No authentication required.
    """
    return {
        "service": "LLM_TYBOO — CPU Edition",
        "version": "1.1.0",
        "models": {
            "llm_14b": "Qwen2.5-14B-Q4_K_M (RAG, analysis, code)",
            "llm_3b": "Qwen2.5-3B-Q4_K_M (chat, simple Q&A)",
            "embeddings": "BAAI/bge-m3 (1024D, multilingual)",
        },
        "endpoints": {
            "auth": ["/token"],
            "rag": ["/rag/query", "/rag/ingest"],
            "agent": ["/agent/general"],
            "system": ["/health", "/info", "/docs"],
        },
    }


# ── RAG endpoints ─────────────────────────────────────────────────────────────

@app.post(
    "/rag/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_auth)],
    tags=["RAG"],
    summary="Query the knowledge base",
)
async def query_rag(request: QueryRequest):
    """
    Ask a question and get an answer grounded in the knowledge base documents.

    How it works:
      1. The question is embedded with BGE-M3
      2. Qdrant finds the top_k most similar document chunks
      3. Chunks below the relevance threshold (0.45) are filtered out
      4. The 14B model generates an answer from the remaining context
      5. If no relevant docs found, the LLM answers from general knowledge

    The response includes the generated answer and (optionally) the source
    document chunks used to generate it, with their relevance scores.
    """
    try:
        result = await rag_system.query(
            question=request.question,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@app.post(
    "/rag/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_auth)],
    tags=["RAG"],
    summary="Add documents to the knowledge base",
)
async def ingest_documents(documents: List[DocumentRequest]):
    """
    Add new documents to the knowledge base for future RAG queries.

    Each document is embedded with BGE-M3 and stored in Qdrant.
    For large files (PDFs, etc.), use ingest.py instead — it handles
    chunking, duplicate detection, and batch processing more efficiently.

    This endpoint is useful for adding small, pre-processed text snippets
    programmatically (e.g., from n8n workflows or other systems).

    Request body example:
      [
        {"text": "Payment is due within 30 days.", "metadata": {"source": "contract.pdf"}},
        {"text": "Late payment incurs 2% interest.", "metadata": {"source": "contract.pdf"}}
      ]
    """
    try:
        docs = [{"text": d.text, "metadata": d.metadata} for d in documents]
        ids = rag_system.ingest_documents(docs)
        return IngestResponse(document_ids=ids, count=len(ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Agent endpoint ────────────────────────────────────────────────────────────

@app.post(
    "/agent/general",
    dependencies=[Depends(require_auth)],
    tags=["Agent"],
    summary="Run the general-purpose automation agent",
)
async def agent_general(request: QueryRequest):
    """
    Run any task through the unrestricted automation agent.

    Unlike /rag/query which only searches documents, this agent can:
      - Call external APIs and webhooks
      - Execute Python code
      - Search the knowledge base (automatically when needed)
      - Send Slack messages and emails
      - Process and transform data

    The agent automatically selects 3B or 14B based on task complexity.
    For simple questions it will respond quickly, for complex multi-step
    tasks it may call multiple tools before returning the final answer.

    Example requests:
      {"question": "What is our refund policy?"}
      {"question": "Fetch https://api.exchangerate.host/EUR and format the rates as a table"}
      {"question": "Analyse the contract I uploaded and list the key obligations"}
    """
    try:
        result = await orchestrator.run_agent(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Use this for quick local testing
    # For production: use the Dockerfile CMD with multiple workers
    uvicorn.run(app, host="0.0.0.0", port=8888)
