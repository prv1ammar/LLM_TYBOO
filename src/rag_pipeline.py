"""
RAG Pipeline using self-hosted embeddings and LLM
"""
import os
from typing import List, Dict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from embeddings import EmbeddingService
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

class SimpleRAG:
    """Simple in-memory RAG system for demonstration"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(
            base_url=f"{LITELLM_URL}/v1",
            api_key=LITELLM_KEY,
        )
        self.model = OpenAIModel(
            'internal-llm',
            provider=provider
        )
        self.agent = Agent(
            self.model,
            system_prompt="""You are an AI assistant for a Moroccan enterprise.
            Answer questions based on the provided context.
            If the context doesn't contain the answer, say so clearly."""
        )
        
        # In-memory document store
        self.documents: List[Dict] = []
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        embedding = self.embedding_service.embed_text(text)
        self.documents.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        })
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant documents"""
        query_embedding = self.embedding_service.embed_text(query)
        
        # Calculate similarities
        similarities = [
            (doc["text"], self.cosine_similarity(query_embedding, doc["embedding"]))
            for doc in self.documents
        ]
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in similarities[:top_k]]
    
    async def query(self, question: str) -> str:
        """Query the RAG system"""
        # Retrieve relevant context
        context_docs = self.retrieve(question)
        context = "\n\n".join(f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs))
        
        # Build prompt with context
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        result = await self.agent.run(prompt)
        return result.data

# Example usage
async def main():
    rag = SimpleRAG()
    
    # Add sample documents about the architecture
    rag.add_document(
        "vLLM is a high-performance inference engine for large language models. "
        "It uses PagedAttention to optimize memory usage and supports tensor parallelism."
    )
    rag.add_document(
        "LiteLLM acts as a unified proxy that provides an OpenAI-compatible API. "
        "It can route requests to multiple backends including vLLM, Anthropic, and OpenAI."
    )
    rag.add_document(
        "Text Embeddings Inference (TEI) is Hugging Face's solution for serving embedding models. "
        "It supports models like BGE-M3 which work well for multilingual content including Arabic."
    )
    rag.add_document(
        "DeepSeek-R1 is a reasoning-focused language model that excels at complex problem-solving. "
        "It provides performance comparable to GPT-4 for enterprise workflows."
    )
    
    # Query the system
    question = "What is vLLM used for?"
    answer = await rag.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
