"""
sdk/llm_client.py — Python SDK for LLM_TYBOO
===============================================
PURPOSE:
  A clean Python client library for integrating LLM_TYBOO into any
  external application without needing to know the internal architecture.

  Instead of writing raw HTTP calls with requests, you import this SDK
  and call simple methods like client.chat() or client.rag_query().

AUTHENTICATION:
  The SDK authenticates using the X-API-Key header.
  The key must match the API_KEY value in your server's .env file.
  Pass it in the constructor or set the API_KEY environment variable.

HOW TO USE:
  # Basic setup
  from sdk.llm_client import LLMBackendClient

  client = LLMBackendClient(
      api_url="http://YOUR_SERVER_IP:8888",
      api_key="your-api-key-from-env"
  )

  # Check service health
  status = client.health()
  print(status)  # {"status": "healthy"}

  # Generate embeddings (BGE-M3, 1024D)
  vectors = client.embed(["Contract clause 1", "Contract clause 2"])
  print(len(vectors[0]))  # 1024

  # Chat completion (auto-routes to 3B or 14B)
  answer = client.chat("What are the key obligations in this contract?")
  print(answer)

  # RAG query — searches knowledge base, answers with citations
  result = client.rag_query("What is the refund policy?", collection="knowledge_base")
  print(result["answer"])
  print(result["sources"])

  # Ingest documents into knowledge base
  client.rag_ingest([
      {"text": "Refunds are accepted within 30 days.", "metadata": {"source": "policy.pdf"}},
  ], collection="knowledge_base")

  # Submit async job (for long operations)
  job_id = client.submit_job("batch_embed", {"texts": ["doc1", "doc2", "doc3"]})
  result = client.get_job_result(job_id, wait=True)  # Blocks until done
  print(result["embeddings"])

  # Check job status without waiting
  status = client.get_job_status(job_id)
  print(status["status"])  # "completed", "running", "pending", "failed"

TIMEOUT:
  All requests use a 120-second timeout by default.
  Override per-call with the requests library if needed.

ERROR HANDLING:
  All methods raise requests.HTTPError on non-2xx responses.
  Wrap calls in try/except if you want custom error handling.
"""

import os
import time
from typing import List, Dict, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMBackendClient:
    """
    Python SDK for the LLM_TYBOO backend API.
    Wraps all HTTP calls in simple, typed methods.

    All methods raise requests.HTTPError if the server returns an error.
    """

    def __init__(self, api_url: str = None, api_key: str = None):
        """
        Initialize the client.

        Args:
            api_url: Base URL of the API server.
                     Defaults to LLM_BACKEND_URL env var, then http://localhost:8888
            api_key: API key for authentication (X-API-Key header).
                     Defaults to API_KEY env var.

        Raises a warning (not an error) if api_key is not provided,
        so you can still call public endpoints like /health.
        """
        self.api_url = (
            api_url
            or os.getenv("LLM_BACKEND_URL")
            or "http://localhost:8888"
        )
        self.api_key = api_key or os.getenv("API_KEY", "")

        if not self.api_key:
            print("Warning: No API key provided. Set API_KEY env var or pass api_key= to constructor.")

        # Every request uses these headers
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    def _request(self, method: str, endpoint: str, timeout: int = 120, **kwargs) -> Dict:
        """
        Internal helper — makes an HTTP request and returns parsed JSON.
        Raises requests.HTTPError on non-2xx status codes.
        """
        url = f"{self.api_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> Dict:
        """
        Check if the API server is running and healthy.

        Returns:
            {"status": "healthy", "service": "..."}

        Usage:
            if client.health()["status"] != "healthy":
                raise RuntimeError("API is down")
        """
        return self._request("GET", "/health")

    def info(self) -> Dict:
        """
        Get platform information: models, version, available endpoints.

        Returns:
            Dict with service name, version, models, and endpoint list.
        """
        return self._request("GET", "/info")

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate BGE-M3 embeddings for a list of texts.

        Args:
            texts: List of strings to embed. Can be any language.

        Returns:
            List of 1024-dimensional float vectors, one per input text.
            Order matches the input: texts[i] → result[i]

        Usage:
            vectors = client.embed(["Hello world", "Bonjour monde"])
            print(len(vectors))    # 2
            print(len(vectors[0])) # 1024
        """
        return self._request("POST", "/api/embeddings", json={"texts": texts})["embeddings"]

    def embed_single(self, text: str) -> List[float]:
        """
        Shorthand for embedding a single text.

        Returns:
            Single 1024-dimensional vector.
        """
        return self.embed([text])[0]

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Send a message and get a response from the auto-routed model.

        The model router automatically selects 3B (fast) or 14B (quality)
        based on the complexity of the message.

        Args:
            message:       The user's message.
            system_prompt: Optional override for the model's system prompt.
                           Use this to give the model a specific persona or task.
            temperature:   0.0 = deterministic, 1.0 = creative (default 0.7)
            max_tokens:    Maximum response length in tokens.

        Returns:
            The model's response as a string.

        Usage:
            response = client.chat("Summarize this contract in 3 bullet points")
        """
        return self._request(
            "POST", "/api/chat",
            json={
                "message": message,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=180  # Chat can take up to 3 minutes on CPU
        )["response"]

    # ── RAG ───────────────────────────────────────────────────────────────────

    def rag_query(self, question: str, collection: str = "default", top_k: int = 3) -> Dict:
        """
        Query a knowledge base collection and get a grounded answer.

        Args:
            question:   The question to ask.
            collection: Which Qdrant collection to search (default "default").
                        Use different collection names for different document sets.
            top_k:      Number of document chunks to retrieve before generating the answer.

        Returns:
            Dict with:
              "answer"              → str, the generated response
              "sources"             → list of source chunks used
              "used_knowledge_base" → bool

        Usage:
            result = client.rag_query(
                "What are the payment terms?",
                collection="legal_contracts"
            )
            print(result["answer"])
            for source in result["sources"]:
                print(f"  [{source['relevance_score']:.2f}] {source['metadata']['source']}")
        """
        return self._request(
            "POST", "/api/rag/query",
            json={"question": question, "collection": collection, "top_k": top_k},
            timeout=180
        )

    def rag_ingest(self, documents: List[Dict], collection: str = "default") -> Dict:
        """
        Ingest documents into a knowledge base collection.

        Args:
            documents:  List of {"text": "...", "metadata": {...}} dicts.
            collection: Target collection name. Created if it doesn't exist.

        Returns:
            {"document_ids": [...], "count": N}

        Usage:
            client.rag_ingest([
                {"text": "Article 1: ...", "metadata": {"source": "contract.pdf", "page": 1}},
                {"text": "Article 2: ...", "metadata": {"source": "contract.pdf", "page": 2}},
            ], collection="legal_contracts")
        """
        return self._request(
            "POST", "/api/rag/ingest",
            json={"documents": documents, "collection": collection}
        )

    # ── Async jobs ────────────────────────────────────────────────────────────

    def submit_job(self, job_type: str, params: Dict, priority: str = "normal") -> str:
        """
        Submit an async job and return its job_id immediately.

        Use this for long operations (large batch embeddings, bulk ingestion)
        that would otherwise time out if done synchronously.

        Args:
            job_type: One of: "batch_embed", "batch_rag_ingest", "analyze_document", "batch_chat"
            params:   Job-type-specific parameters dict.
            priority: "low", "normal", or "high"

        Returns:
            job_id string — use this to poll get_job_status() or get_job_result()

        Usage:
            job_id = client.submit_job("batch_embed", {"texts": ["a", "b", "c"]})
            print(f"Job submitted: {job_id}")
        """
        return self._request(
            "POST", "/api/jobs",
            json={"job_type": job_type, "params": params, "priority": priority}
        )["job_id"]

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get the current status and progress of a job.

        Returns:
            Dict with keys: job_id, status, progress (0-100),
            result (when complete), error (when failed),
            created_at, started_at, completed_at

        Usage:
            status = client.get_job_status(job_id)
            print(status["status"])    # "pending", "running", "completed", "failed"
            print(status["progress"])  # 0-100
        """
        return self._request("GET", f"/api/jobs/{job_id}")

    def get_job_result(self, job_id: str, wait: bool = True, timeout: int = 300) -> Any:
        """
        Get the result of a completed job.

        Args:
            job_id:  The job ID returned by submit_job().
            wait:    If True (default), block until the job completes or timeout.
                     If False, return immediately with current result (may be None).
            timeout: Maximum seconds to wait when wait=True (default 300s = 5 minutes).

        Returns:
            The job's result value (type depends on job_type).

        Raises:
            Exception:    If the job failed, raises with the error message.
            TimeoutError: If wait=True and job doesn't complete within timeout.

        Usage:
            # Wait for completion
            result = client.get_job_result(job_id, wait=True)
            print(result["embeddings"])

            # Non-blocking check
            result = client.get_job_result(job_id, wait=False)
            if result is None:
                print("Job still running...")
        """
        if not wait:
            return self.get_job_status(job_id).get("result")

        # Polling loop — checks every 2 seconds
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_job_status(job_id)
            if status["status"] == "completed":
                return status["result"]
            if status["status"] == "failed":
                raise Exception(f"Job {job_id} failed: {status.get('error', 'unknown error')}")
            if status["status"] == "cancelled":
                raise Exception(f"Job {job_id} was cancelled")
            time.sleep(2)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def cancel_job(self, job_id: str) -> Dict:
        """
        Cancel a pending or running job.

        Returns:
            {"message": "Job cancelled", "job_id": "..."}
        """
        return self._request("DELETE", f"/api/jobs/{job_id}")


# ── Module-level convenience functions ───────────────────────────────────────
# For quick one-off calls without creating a client instance.

def embed(texts: List[str], api_url: str = None, api_key: str = None) -> List[List[float]]:
    """Quick embed without instantiating a client manually."""
    return LLMBackendClient(api_url, api_key).embed(texts)


def chat(message: str, api_url: str = None, api_key: str = None) -> str:
    """Quick chat call without instantiating a client manually."""
    return LLMBackendClient(api_url, api_key).chat(message)


def rag_query(question: str, collection: str = "default",
              api_url: str = None, api_key: str = None) -> Dict:
    """Quick RAG query without instantiating a client manually."""
    return LLMBackendClient(api_url, api_key).rag_query(question, collection)
