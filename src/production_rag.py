"""
Production RAG system with vector database
Combines VectorStore with LLM for advanced question answering
"""
import os
from typing import List, Dict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-1234")

class ProductionRAG:
    """Production-ready RAG system with vector database"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.vector_store = VectorStore(collection_name=collection_name)
        
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(
            base_url=f"{LITELLM_URL}/v1",
            api_key=LITELLM_KEY,
        )
        
        model = OpenAIModel(
            'internal-llm',
            provider=provider
        )
        
        self.agent = Agent(
            model,
            system_prompt="""You are an AI assistant for Moroccan enterprises.
            Answer questions based ONLY on the provided context.
            If the context doesn't contain enough information, say so clearly.
            Always cite which document you're referencing."""
        )
    
    def ingest_documents(self, documents: List[Dict[str, str]]) -> List[str]:
        """
        Ingest documents into the knowledge base
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        
        Returns:
            List of document IDs
        """
        return self.vector_store.add_documents(documents)
    
    async def query(self, question: str, top_k: int = 3, include_sources: bool = True) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            top_k: Number of relevant documents to retrieve
            include_sources: Whether to include source documents in response
        
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Retrieve relevant documents
        results = self.vector_store.search(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "I don't have any relevant information to answer this question.",
                "sources": []
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Document {i} (relevance: {result['score']:.2f}):\n{result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Context from knowledge base:

{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Reference specific documents when applicable."""
        
        # Get answer from LLM
        response = await self.agent.run(prompt)
        
        result = {"answer": response.data}
        
        if include_sources:
            result["sources"] = [
                {
                    "text": r['text'],
                    "metadata": r['metadata'],
                    "relevance_score": r['score']
                }
                for r in results
            ]
        
        return result

# Example usage
async def main():
    print("🚀 Production RAG System Demo\n")
    print("=" * 70)
    
    # Initialize RAG system
    rag = ProductionRAG(collection_name="moroccan_enterprise_kb")
    
    # Ingest knowledge base
    print("\n📚 Ingesting knowledge base...")
    documents = [
        {
            "text": """Morocco has established itself as a leader in renewable energy in Africa. 
            The country aims to generate 52% of its electricity from renewable sources by 2030. 
            Major projects include the Noor Ouarzazate Solar Complex, one of the world's largest 
            concentrated solar power plants.""",
            "metadata": {"topic": "energy", "source": "government_report_2024"}
        },
        {
            "text": """The Moroccan banking sector is well-regulated and stable. Major banks include 
            Attijariwafa Bank, Banque Populaire, and BMCE Bank. The sector is increasingly 
            adopting digital banking solutions and fintech innovations.""",
            "metadata": {"topic": "finance", "source": "banking_overview_2024"}
        },
        {
            "text": """Morocco's automotive industry has grown significantly, with major manufacturers 
            like Renault and PSA establishing production facilities. The sector employs over 
            220,000 people and exports vehicles to European and African markets.""",
            "metadata": {"topic": "manufacturing", "source": "industry_report_2024"}
        },
        {
            "text": """Digital transformation is a priority for Moroccan businesses. The government's 
            Digital Morocco 2020 strategy has been succeeded by Morocco Digital 2030, focusing 
            on AI, cloud computing, and cybersecurity.""",
            "metadata": {"topic": "technology", "source": "digital_strategy_2024"}
        },
        {
            "text": """Casablanca Finance City (CFC) is a financial hub offering tax incentives and 
            streamlined regulations for international companies. It hosts over 200 companies 
            from various sectors including banking, insurance, and consulting.""",
            "metadata": {"topic": "finance", "source": "cfc_overview_2024"}
        },
        {
            "text": """Morocco's tourism sector is vital to the economy, contributing approximately 
            7% of GDP. Popular destinations include Marrakech, Fez, Casablanca, and the 
            coastal cities. The sector is recovering strongly post-pandemic.""",
            "metadata": {"topic": "tourism", "source": "tourism_stats_2024"}
        }
    ]
    
    doc_ids = rag.ingest_documents(documents)
    print(f"✅ Ingested {len(doc_ids)} documents\n")
    
    # Test queries
    print("=" * 70)
    print("\n🔍 Testing RAG System with Sample Queries\n")
    
    queries = [
        "What renewable energy projects exist in Morocco?",
        "Tell me about the banking sector in Morocco",
        "What is Casablanca Finance City?",
        "How important is the automotive industry?",
    ]
    
    for i, question in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {question}")
        print('='*70)
        
        result = await rag.query(question, top_k=2, include_sources=True)
        
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        print("📄 Sources:")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. [Relevance: {source['relevance_score']:.3f}] "
                  f"{source['metadata'].get('source', 'Unknown')}")
            print(f"     {source['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
