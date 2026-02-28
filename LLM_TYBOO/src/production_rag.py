"""
production_rag.py — Retrieval-Augmented Generation Pipeline
=============================================================
PURPOSE:
  Answers user questions by finding relevant documents first,
  then using the LLM to synthesize a response from that context.
  This is better than asking the LLM directly because:
    - Answers are grounded in YOUR documents (not just training data)
    - Sources are cited so users can verify answers
    - The LLM can answer even if it wasn't trained on your domain

HOW RAG WORKS (step by step):
  1. User sends a question
  2. The question is embedded into a 1024D vector using BGE-M3
  3. Qdrant finds the top-k most similar document chunks
  4. Chunks with score < 0.45 (relevance threshold) are filtered out
  5. If relevant chunks exist → prompt is built with context + question
     If no relevant chunks  → LLM answers from general knowledge (no refusal)
  6. LLM (14B) generates a response
  7. Response + sources are returned to the caller

RELEVANCE THRESHOLD (0.45):
  Cosine similarity ranges from 0.0 (unrelated) to 1.0 (identical).
  At 0.45, we only keep chunks that are reasonably related to the question.
  If you get too many "no documents found" responses, lower this value.
  If you get irrelevant context being used, raise this value.

NO REFUSAL POLICY:
  If no relevant documents are found, the LLM still answers the question
  using its general knowledge. It never says "I don't have information
  about this." This makes the system useful even for off-topic queries.

HOW TO USE:
  from production_rag import ProductionRAG

  rag = ProductionRAG(collection_name="knowledge_base")

  # Add documents to the knowledge base
  rag.ingest_documents([
      {"text": "Our refund policy allows returns within 30 days.",
       "metadata": {"source": "policy.pdf"}}
  ])

  # Query the knowledge base
  result = await rag.query("What is the refund policy?")
  print(result["answer"])           # LLM-generated answer
  print(result["sources"])          # list of source chunks used
  print(result["used_knowledge_base"])  # True if KB was used, False if general knowledge
"""

import os
from typing import List, Dict
from pydantic_ai import Agent
from vector_store import VectorStore
from model_router import get_model_for_rag
from dotenv import load_dotenv

load_dotenv()

# Documents with similarity score below this threshold are not sent to the LLM.
# Lower = more permissive (includes less relevant docs)
# Higher = more strict (only very relevant docs are used)
RELEVANCE_THRESHOLD = 0.45


class ProductionRAG:
    """
    Production-ready RAG system with knowledge base + general knowledge fallback.

    Can handle any question in French, Arabic, Darija, or English.
    Always uses 14B model for best answer quality.
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        """
        Initialize RAG with a specific Qdrant collection.

        Args:
            collection_name: Which Qdrant collection to search.
                             You can run multiple RAG instances with different
                             collections for different document sets.
                             Example: "legal_docs", "hr_policies", "technical_manuals"
        """
        self.vector_store = VectorStore(collection_name=collection_name)

        # RAG always uses 14B — never route RAG queries to 3B
        # Accuracy matters more than speed when answering from documents
        model = get_model_for_rag()

        self.agent = Agent(
            model,
            system_prompt="""You are a powerful general-purpose AI assistant.
When context documents are provided, base your answer on them and cite the source documents.
When no relevant context is found, answer from your general knowledge — never refuse to answer.
Be precise, structured, and complete in your responses.
Always respond in the same language the user used in their question."""
        )

    def ingest_documents(self, documents: List[Dict]) -> List[str]:
        """
        Add documents to the knowledge base.

        This is a thin wrapper around VectorStore.add_documents().
        See vector_store.py for full details on the format.

        Args:
            documents: List of {"text": "...", "metadata": {...}} dicts.

        Returns:
            List of UUID strings — one per document stored.

        Usage:
            ids = rag.ingest_documents([
                {"text": "Article 3: ...", "metadata": {"source": "contract.pdf", "page": 1}},
            ])
        """
        return self.vector_store.add_documents(documents)

    async def query(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question:           The user's question (any language).
            top_k:              How many document chunks to retrieve from Qdrant.
                                3 is usually enough; increase for broad topics.
            include_sources:    If True, include the source chunks in the response.
                                Set to False to reduce response size.
            relevance_threshold: Minimum cosine similarity score to include a chunk.
                                 Default is 0.45.

        Returns:
            Dict with these keys:
              "answer"              → str, the LLM-generated response
              "sources"             → list of source dicts (if include_sources=True)
              "used_knowledge_base" → bool, True if KB docs were used

            Each source dict:
              {
                  "text": "original chunk text",
                  "metadata": {"source": "file.pdf", ...},
                  "relevance_score": 0.83
              }

        How the prompt is built:
            WITH relevant docs:
              "Context: [doc1] [doc2] ... Question: ... Answer based on context."

            WITHOUT relevant docs:
              "No KB docs found. Question: ... Answer from general knowledge."
        """
        # Step 1: Retrieve candidate documents from Qdrant
        candidates = self.vector_store.search(question, top_k=top_k)

        # Step 2: Filter by relevance threshold
        relevant = [r for r in candidates if r["score"] >= relevance_threshold]

        # Step 3: Build the prompt based on what was found
        if relevant:
            # Build context block with source labels for citation
            context_parts = [
                f"[Document {i} | score {r['score']:.2f} | source: {r['metadata'].get('source', 'unknown')}]\n{r['text']}"
                for i, r in enumerate(relevant, 1)
            ]
            context = "\n\n".join(context_parts)
            prompt = (
                f"Context from knowledge base:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer based on the context above. Cite which document(s) you used."
            )
        else:
            # No relevant documents found — ask LLM to answer from general knowledge
            # This prevents the system from refusing to answer off-topic questions
            prompt = (
                f"No relevant documents were found in the knowledge base for this question.\n\n"
                f"Question: {question}\n\n"
                f"Answer using your general knowledge. Be thorough and helpful."
            )

        # Step 4: Generate the answer with the LLM
        response = await self.agent.run(prompt)

        # Step 5: Build the response dict
        result = {"answer": response.output}

        if include_sources:
            result["sources"] = [
                {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "relevance_score": r["score"]
                }
                for r in relevant
            ]
            # Lets the caller know whether the answer came from the KB or general knowledge
            result["used_knowledge_base"] = len(relevant) > 0

        return result
