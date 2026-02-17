"""
Production-ready vector database integration using Qdrant
For scalable RAG systems
"""
import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings import EmbeddingService
from dotenv import load_dotenv
import uuid

load_dotenv()

class VectorStore:
    """Production vector store using Qdrant"""
    
    def __init__(self, collection_name: str = "documents"):
        # For development: in-memory Qdrant
        # For production: connect to Qdrant server
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # If QDRANT_URL is not set to a server, use in-memory
        if not qdrant_url or qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)
            
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize the vector collection"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # BGE-M3 produces 1024-dimensional embeddings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, str]]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Extract texts for batch embedding
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings in batch
        embeddings = self.embedding_service.embed_batch(texts)
        
        # Prepare points for Qdrant
        points = []
        doc_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": doc['text'],
                    "metadata": doc.get('metadata', {})
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Added {len(points)} documents to vector store")
        return doc_ids
    
    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_dict
        )
        
        # Format results
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "text": hit.payload['text'],
                "metadata": hit.payload.get('metadata', {}),
                "score": hit.score
            })
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"🗑️  Deleted collection: {self.collection_name}")

# Example usage
async def main():
    print("🔍 Vector Store Example\n")
    
    # Initialize vector store
    vector_store = VectorStore(collection_name="moroccan_business_docs")
    
    # Sample documents about Moroccan business context
    documents = [
        {
            "text": "Morocco's digital transformation is accelerating with government support for AI and cloud computing initiatives.",
            "metadata": {"category": "technology", "year": 2024}
        },
        {
            "text": "Casablanca Finance City is becoming a major hub for fintech and financial services in Africa.",
            "metadata": {"category": "finance", "year": 2024}
        },
        {
            "text": "The Moroccan government is investing heavily in renewable energy, particularly solar and wind power.",
            "metadata": {"category": "energy", "year": 2024}
        },
        {
            "text": "E-commerce is growing rapidly in Morocco, with local platforms competing with international players.",
            "metadata": {"category": "retail", "year": 2024}
        },
        {
            "text": "Morocco's automotive industry is one of the largest in Africa, exporting to European markets.",
            "metadata": {"category": "manufacturing", "year": 2024}
        }
    ]
    
    # Add documents
    doc_ids = vector_store.add_documents(documents)
    print(f"Document IDs: {doc_ids[:2]}...\n")
    
    # Search examples
    queries = [
        "Tell me about technology and AI in Morocco",
        "What are the main industries in Morocco?",
        "Financial services and banking"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        results = vector_store.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [Score: {result['score']:.3f}] {result['text'][:80]}...")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
