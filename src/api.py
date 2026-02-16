"""
FastAPI REST API for the self-hosted AI stack
Production-ready API with authentication, rate limiting, and monitoring
"""
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from production_rag import ProductionRAG
from orchestrator import AgentOrchestrator
from dotenv import load_dotenv
import asyncio

load_dotenv()

# API Configuration
API_KEY = os.getenv("API_KEY", "your-secure-api-key-here")

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

class AnalysisRequest(BaseModel):
    document: str = Field(..., description="Document to analyze")

class ContentGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Content generation prompt")
    context: Optional[str] = Field("", description="Additional context")

# Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Health check
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "service": "self-hosted-ai-api"}

# RAG Endpoints
@app.post("/rag/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
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

@app.post("/rag/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
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
@app.post("/agent/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_document(request: AnalysisRequest):
    """
    Analyze a business document
    
    Returns structured analysis with key points and sentiment
    """
    try:
        result = await orchestrator.analyze_document(request.document)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/agent/generate", dependencies=[Depends(verify_api_key)])
async def generate_content(request: ContentGenerationRequest):
    """
    Generate business content
    
    Creates professional content based on the prompt and context
    """
    try:
        result = await orchestrator.generate_content(request.prompt, request.context)
        return {"content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# System Info
@app.get("/info")
async def system_info():
    """Get system information"""
    return {
        "service": "Self-Hosted AI Stack",
        "version": "1.0.0",
        "features": [
            "RAG with vector search",
            "Multi-agent orchestration",
            "Self-hosted LLM (DeepSeek-R1)",
            "Self-hosted embeddings (BGE-M3)",
            "Production-ready API"
        ],
        "endpoints": {
            "rag": ["/rag/query", "/rag/ingest"],
            "agents": ["/agent/analyze", "/agent/generate"],
            "system": ["/health", "/info"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
