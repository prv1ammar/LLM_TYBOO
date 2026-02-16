"""
Embedding service using self-hosted TEI via LiteLLM
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

class EmbeddingService:
    """Self-hosted embedding service using BGE-M3 via TEI"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url=f"{LITELLM_URL}/v1",
            api_key=LITELLM_KEY
        )
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for a single text"""
        try:
            response = self.client.embeddings.create(
                model="internal-embedding",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}. Is Docker running? Check LiteLLM/TEI containers.")
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model="internal-embedding",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}. Is Docker running? Check LiteLLM/TEI containers.")

# Example usage
if __name__ == "__main__":
    service = EmbeddingService()
    
    # Test single embedding
    text = "This is a test document about AI infrastructure in Morocco"
    embedding = service.embed_text(text)
    print(f"Generated embedding with {len(embedding)} dimensions")
    
    # Test batch embeddings
    texts = [
        "DeepSeek-R1 is a reasoning model",
        "BGE-M3 supports multilingual embeddings",
        "vLLM provides fast inference"
    ]
    embeddings = service.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")
