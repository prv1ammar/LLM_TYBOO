"""
api.py — FastAPI REST API
===========================
Users are stored in PostgreSQL (tyboo_users table).
No hardcoded credentials — add users via POST /admin/users or the dashboard.

ADMIN ENDPOINTS (admin role required):
  GET    /admin/users                       — List all users
  POST   /admin/users                       — Create a new user
  DELETE /admin/users/{username}            — Delete a user
  PATCH  /admin/users/{username}/password   — Change password
  PATCH  /admin/users/{username}/active     — Enable / disable

DEFAULT ADMIN: seeded from ADMIN_USERNAME + ADMIN_PASSWORD in .env on first startup.
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

app = FastAPI(
    title="LLM_TYBOO API",
    description="Self-Hosted AI Platform — CPU Edition (Qwen2.5 14B + 3B)",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = ProductionRAG(collection_name="enterprise_kb")
orchestrator = AgentOrchestrator()


@app.on_event("startup")
async def startup():
    try:
        init_users_table()
    except Exception as e:
        print(f"[WARN] Could not initialize users table: {e}")


# ── Models ────────────────────────────────────────────────────────────────────
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

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field("user")

class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)

class SetActiveRequest(BaseModel):
    is_active: bool


# ── Auth dependencies ─────────────────────────────────────────────────────────
async def require_auth(current_user: dict = Depends(get_current_user)):
    return current_user

async def require_admin(current_user: dict = Depends(get_current_user)):
    user = get_user(current_user["username"])
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Admin role required")
    return current_user


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/token", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


# ── System ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "service": "llm-tyboo-api"}

@app.get("/info", tags=["System"])
async def info():
    return {
        "service": "LLM_TYBOO — CPU Edition", "version": "1.2.0",
        "models": {"llm_14b": "Qwen2.5-14B-Q4_K_M", "llm_3b": "Qwen2.5-3B-Q4_K_M",
                   "embeddings": "BAAI/bge-m3 (1024D)"},
        "endpoints": {"auth": ["/token"], "rag": ["/rag/query", "/rag/ingest"],
                      "agent": ["/agent/general"], "admin": ["/admin/users"],
                      "system": ["/health", "/info", "/docs"]},
    }


# ── RAG ───────────────────────────────────────────────────────────────────────
@app.post("/rag/query", response_model=QueryResponse, dependencies=[Depends(require_auth)], tags=["RAG"])
async def query_rag(request: QueryRequest):
    try:
        result = await rag_system.query(question=request.question, top_k=request.top_k,
                                         include_sources=request.include_sources)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(require_auth)], tags=["RAG"])
async def ingest_documents(documents: List[DocumentRequest]):
    try:
        docs = [{"text": d.text, "metadata": d.metadata} for d in documents]
        ids = rag_system.ingest_documents(docs)
        return IngestResponse(document_ids=ids, count=len(ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Agent ─────────────────────────────────────────────────────────────────────
@app.post("/agent/general", dependencies=[Depends(require_auth)], tags=["Agent"])
async def agent_general(request: QueryRequest):
    try:
        result = await orchestrator.run_agent(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


# ── Admin: User Management ────────────────────────────────────────────────────
@app.get("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin — Users"])
async def admin_list_users():
    """List all users (no passwords). Requires admin role."""
    try:
        return {"users": list_users()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users", dependencies=[Depends(require_admin)], tags=["Admin — Users"], status_code=201)
async def admin_create_user(body: CreateUserRequest):
    """
    Create a new user.
    Body: { "username": "ammar", "password": "ammar@2025", "role": "user" }
    Requires admin role.
    """
    if body.role not in ["admin", "user"]:
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
    try:
        user = create_user(body.username, body.password, body.role)
        return {"message": f"User '{body.username}' created", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/users/{username}", tags=["Admin — Users"])
async def admin_delete_user(username: str, current_user: dict = Depends(require_admin)):
    """Delete a user. Cannot delete yourself. Requires admin role."""
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    try:
        if not delete_user(username):
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        return {"message": f"User '{username}' deleted"}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.patch("/admin/users/{username}/password", dependencies=[Depends(require_admin)], tags=["Admin — Users"])
async def admin_change_password(username: str, body: ChangePasswordRequest):
    """Change a user's password. Requires admin role."""
    try:
        if not update_password(username, body.new_password):
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        return {"message": f"Password for '{username}' updated"}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.patch("/admin/users/{username}/active", tags=["Admin — Users"])
async def admin_set_active(username: str, body: SetActiveRequest,
                            current_user: dict = Depends(require_admin)):
    """Enable or disable a user. Requires admin role."""
    if username == current_user["username"] and not body.is_active:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")
    try:
        if not set_user_active(username, body.is_active):
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        state = "enabled" if body.is_active else "disabled"
        return {"message": f"User '{username}' {state}"}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
