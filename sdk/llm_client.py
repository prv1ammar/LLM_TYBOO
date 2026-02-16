"""
Reusable LLM Backend Service - Core SDK
This module provides a clean interface for client applications to interact with the LLM backend
"""
import os
from typing import List, Dict, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMBackendClient:
    """
    Client SDK for the LLM Backend Service
    
    Usage:
        client = LLMBackendClient(api_url="https://your-llm-backend.com")
        
        # Generate embeddings
        embeddings = client.embed(["text1", "text2"])
        
        # Chat completion
        response = client.chat("What is Morocco's capital?")
        
        # Submit async job
        job_id = client.submit_job("analyze_document", {"text": "..."})
        result = client.get_job_result(job_id)
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or os.getenv("LLM_BACKEND_URL", "http://localhost:8888")
        self.api_url = api_url or os.getenv("LLM_BACKEND_URL", "http://localhost:8888")
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("LLM_BACKEND_API_KEY")
        if not self.api_key:
             # Try to find .env file in parent directories if not loaded
             if os.path.exists(".env"):
                 from dotenv import load_dotenv
                 load_dotenv()
                 self.api_key = os.getenv("API_KEY")

        if not self.api_key:
            print("⚠️ Warning: No API Key provided in constructor or environment variable API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Internal method for making HTTP requests"""
        url = f"{self.api_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    # Embedding Methods
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        response = self._request("POST", "/api/embeddings", json={"texts": texts})
        return response["embeddings"]
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed([text])[0]
    
    # Chat/Completion Methods
    def chat(
        self, 
        message: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Send a chat message and get a response
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Assistant's response
        """
        payload = {
            "message": message,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = self._request("POST", "/api/chat", json=payload)
        return response["response"]
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Text completion
        
        Args:
            prompt: Prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completion text
        """
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = self._request("POST", "/api/complete", json=payload)
        return response["completion"]
    
    # RAG Methods
    def rag_query(
        self,
        question: str,
        collection: str = "default",
        top_k: int = 3
    ) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            collection: Vector collection name
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        payload = {
            "question": question,
            "collection": collection,
            "top_k": top_k
        }
        return self._request("POST", "/api/rag/query", json=payload)
    
    def rag_ingest(
        self,
        documents: List[Dict[str, Any]],
        collection: str = "default"
    ) -> Dict:
        """
        Ingest documents into RAG system
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
            collection: Vector collection name
            
        Returns:
            Dict with 'document_ids' and 'count'
        """
        payload = {
            "documents": documents,
            "collection": collection
        }
        return self._request("POST", "/api/rag/ingest", json=payload)
    
    # Async Job Methods
    def submit_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        priority: str = "normal"
    ) -> str:
        """
        Submit an async job
        
        Args:
            job_type: Type of job (e.g., 'analyze_document', 'batch_embed')
            params: Job parameters
            priority: Job priority ('low', 'normal', 'high')
            
        Returns:
            Job ID
        """
        payload = {
            "job_type": job_type,
            "params": params,
            "priority": priority
        }
        response = self._request("POST", "/api/jobs", json=payload)
        return response["job_id"]
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get job status
        
        Returns:
            Dict with 'status', 'progress', 'result' (if complete)
        """
        return self._request("GET", f"/api/jobs/{job_id}")
    
    def get_job_result(self, job_id: str, wait: bool = True, timeout: int = 300) -> Any:
        """
        Get job result (optionally wait for completion)
        
        Args:
            job_id: Job ID
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            Job result
        """
        if wait:
            import time
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self.get_job_status(job_id)
                if status["status"] == "completed":
                    return status["result"]
                elif status["status"] == "failed":
                    raise Exception(f"Job failed: {status.get('error')}")
                time.sleep(2)
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
        else:
            status = self.get_job_status(job_id)
            return status.get("result")
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a running job"""
        return self._request("DELETE", f"/api/jobs/{job_id}")
    
    # Agent Methods
    def analyze_document(self, document: str) -> Dict:
        """
        Analyze a business document
        
        Returns:
            Dict with 'summary', 'key_points', 'sentiment', 'confidence'
        """
        payload = {"document": document}
        return self._request("POST", "/api/agent/analyze", json=payload)
    
    def generate_content(self, prompt: str, context: str = "") -> str:
        """
        Generate business content
        
        Returns:
            Generated content
        """
        payload = {"prompt": prompt, "context": context}
        response = self._request("POST", "/api/agent/generate", json=payload)
        return response["content"]
    
    # Health & Info
    def health(self) -> Dict:
        """Check service health"""
        return self._request("GET", "/health")
    
    def info(self) -> Dict:
        """Get service information"""
        return self._request("GET", "/info")


# Convenience functions for quick usage
def embed(texts: List[str], api_url: str = None, api_key: str = None) -> List[List[float]]:
    """Quick embedding function"""
    client = LLMBackendClient(api_url, api_key)
    return client.embed(texts)

def chat(message: str, api_url: str = None, api_key: str = None) -> str:
    """Quick chat function"""
    client = LLMBackendClient(api_url, api_key)
    return client.chat(message)

def rag_query(question: str, collection: str = "default", api_url: str = None, api_key: str = None) -> Dict:
    """Quick RAG query function"""
    client = LLMBackendClient(api_url, api_key)
    return client.rag_query(question, collection)


# Example usage
if __name__ == "__main__":
    # Initialize client (will try to load API_KEY from .env)
    # Ensure you have a .env file with API_KEY=...
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("⚠️  No API_KEY in environment. Please check your .env file.")
        # Fallback for testing if running without .env loaded
        api_key = "8f7a9c2e-1b4d-4e91-a9f1-7c3e5b2d1a90" 

    client = LLMBackendClient(
        api_url="http://localhost:8888",
        api_key=api_key
    )
    
    # Example 1: Generate embeddings
    print("Example 1: Embeddings")
    embeddings = client.embed(["Hello world", "Morocco is beautiful"])
    print(f"Generated {len(embeddings)} embeddings")
    
    # Example 2: Chat
    print("\nExample 2: Chat")
    response = client.chat("What is the capital of Morocco?")
    print(f"Response: {response}")
    
    # Example 3: RAG Query
    print("\nExample 3: RAG Query")
    result = client.rag_query("Tell me about renewable energy in Morocco")
    print(f"Answer: {result['answer']}")
    
    # Example 4: Async Job
    print("\nExample 4: Async Job")
    job_id = client.submit_job(
        job_type="batch_embed",
        params={"texts": ["text1", "text2", "text3"]}
    )
    print(f"Job submitted: {job_id}")
    result = client.get_job_result(job_id, wait=True)
    print(f"Job result: {result}")
