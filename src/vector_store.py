"""
vector_store.py — Qdrant Vector Database Integration
======================================================
PURPOSE:
  Stores document embeddings and performs semantic similarity search.
  This is the core of the RAG retrieval step.

HOW IT WORKS:
  1. add_documents() — takes raw text chunks, embeds them using BGE-M3,
     and stores the (vector, text, metadata) triplets in Qdrant.

  2. search() — takes a query string, embeds it, then asks Qdrant
     to find the most similar stored vectors using cosine similarity.
     Returns the top-k most relevant document chunks.

VECTOR DIMENSIONS:
  BGE-M3 produces 1024-dimensional vectors.
  The Qdrant collection is created with size=1024 and Distance.COSINE.

DOCUMENT IDs:
  Each document gets a random UUID as its ID in Qdrant.
  add_documents() returns these IDs so you can reference or delete
  specific documents later if needed.

CONNECTION MODES:
  - Docker mode (default): QDRANT_URL=http://qdrant:6333
  - Local dev mode:        QDRANT_URL=http://localhost:6333
  - In-memory mode:        QDRANT_URL=:memory:  (data lost on restart, good for testing)

HOW TO USE:
  from vector_store import VectorStore

  store = VectorStore(collection_name="my_documents")

  # Add documents to the store
  docs = [
      {"text": "Payment is due within 30 days.", "metadata": {"source": "contract_01.pdf"}},
      {"text": "Late payment incurs 2% monthly interest.", "metadata": {"source": "contract_01.pdf"}},
  ]
  ids = store.add_documents(docs)
  print(ids)  # ["uuid-1", "uuid-2"]

  # Search for similar documents
  results = store.search("What are the payment terms?", top_k=3)
  for r in results:
      print(r["score"], r["text"])
      # 0.91  "Payment is due within 30 days."
"""

import os
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings import EmbeddingService
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    Wraps Qdrant to provide simple add/search operations for document chunks.

    Each instance is tied to one collection_name.
    Multiple collections can exist in the same Qdrant instance — for example
    "knowledge_base" for general docs and "contracts" for legal documents.
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        """
        Connect to Qdrant and ensure the collection exists.

        If the collection doesn't exist yet, it's created automatically
        with the correct vector size (1024) and distance metric (cosine).

        Args:
            collection_name: Name of the Qdrant collection to use.
                             Created automatically if it doesn't exist.
        """
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        # In-memory mode for development/testing — no Docker needed
        # All data is lost when the process exits
        if not qdrant_url or qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)

        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

        # Create collection if it doesn't exist yet
        self._initialize_collection()

    def _initialize_collection(self):
        """
        Create the Qdrant collection if it doesn't already exist.

        Collection settings:
          - size=1024: matches BGE-M3 output dimensions
          - distance=COSINE: measures angle between vectors, good for text similarity
            (range: 0.0 = unrelated, 1.0 = identical meaning)
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {self.collection_name}")

    def add_documents(self, documents: List[Dict]) -> List[str]:
        """
        Embed and store a list of document chunks in Qdrant.

        Each document dict must have a "text" key.
        Optionally include a "metadata" dict with source, page, date, etc.

        Args:
            documents: List of dicts like:
                [
                    {"text": "...", "metadata": {"source": "file.pdf", "page": 1}},
                    {"text": "...", "metadata": {"source": "file.pdf", "page": 2}},
                ]

        Returns:
            List of UUIDs assigned to each document in Qdrant.
            Store these if you need to delete specific documents later.

        How it works internally:
            1. Extract all text strings from the input
            2. Send them all to BGE-M3 in one batch (efficient)
            3. Pair each text with its vector and create a PointStruct
            4. Upload all points to Qdrant in a single upsert call
        """
        if not documents:
            return []

        # Embed all texts in a single batch call — faster than one by one
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_service.embed_batch(texts)

        points = []
        doc_ids = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)

            # PointStruct is what Qdrant stores:
            #   id     → UUID string
            #   vector → the 1024-float embedding
            #   payload → the original text + metadata (returned at search time)
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {})
                }
            ))

        # upsert: insert new points or update existing ones with the same ID
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Stored {len(points)} document chunks in collection '{self.collection_name}'")
        return doc_ids

    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Find the most semantically similar documents to the query.

        Args:
            query: Natural language question or search string.
                   Will be embedded using BGE-M3 before searching.
            top_k: Number of results to return (default 5, max ~20 is useful).
            filter_dict: Optional Qdrant filter to restrict search to specific
                         metadata values. Example:
                         {"must": [{"key": "metadata.source", "match": {"value": "contract.pdf"}}]}

        Returns:
            List of dicts sorted by similarity score (highest first):
            [
                {
                    "id": "uuid-string",
                    "text": "original document chunk text",
                    "metadata": {"source": "file.pdf", "page": 3},
                    "score": 0.87   # cosine similarity, 0.0 to 1.0
                },
                ...
            ]

        Note:
            The production_rag.py layer filters results to score >= 0.45
            (the relevance threshold). Results below that are considered
            too dissimilar and are dropped before sending to the LLM.
        """
        # Embed the query using the same model used at ingestion time
        query_vector = self.embedding_service.embed_text(query)

        # query_points returns the top_k most similar vectors in the collection
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=filter_dict
        ).points

        return [
            {
                "id": hit.id,
                "text": hit.payload["text"],
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score   # cosine similarity score
            }
            for hit in results
        ]

    def delete_collection(self):
        """
        Permanently delete this collection and all its documents from Qdrant.
        Use with caution — this cannot be undone.

        Usage:
            store = VectorStore("old_collection")
            store.delete_collection()
        """
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
