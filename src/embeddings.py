"""
embeddings.py — Local Embedding Service
=========================================
PURPOSE:
  Converts text into numerical vectors (embeddings) that represent
  semantic meaning. Similar texts produce similar vectors, which allows
  Qdrant to find relevant documents using vector similarity search.

MODEL USED:
  BAAI/bge-m3 — 1024-dimensional vectors
  - Multilingual: French, Arabic, Darija, English all supported
  - Free and runs fully locally (no API calls, no data leaves the server)
  - Downloaded once (~570MB) and cached in ~/.cache/huggingface/
  - After first download, works completely offline

HOW EMBEDDINGS WORK IN THIS PROJECT:
  1. At ingestion time: each document chunk is embedded and stored in Qdrant
  2. At query time: the user's question is embedded and compared against
     stored vectors using cosine similarity to find relevant documents

LAZY LOADING:
  The model is only loaded into memory on the first call to embed_text()
  or embed_batch(). This avoids wasting RAM if embeddings aren't needed yet.
  Once loaded, the model stays in memory for the lifetime of the process.

HOW TO USE:
  from embeddings import EmbeddingService

  service = EmbeddingService()

  # Embed a single piece of text
  vector = service.embed_text("This is a contract clause about penalties")
  print(len(vector))  # 1024

  # Embed multiple texts at once (faster than calling embed_text in a loop)
  vectors = service.embed_batch(["text one", "text two", "text three"])
  print(len(vectors))    # 3
  print(len(vectors[0])) # 1024

TO TEST THIS FILE:
  python embeddings.py
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Module-level variable — holds the model after first load
# None means model hasn't been loaded yet (lazy initialization)
_model = None


def _get_model():
    """
    Load the BGE-M3 model on first call, then reuse it on subsequent calls.

    Why lazy loading?
    - The model takes a few seconds to load (~2GB into RAM)
    - If your code imports embeddings.py but doesn't call embed_text() yet,
      you don't want to waste time loading the model prematurely
    - First call is slow, every subsequent call is instant (model stays in RAM)
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        print(f"Loading embedding model: {model_name} ...")
        print("First run downloads ~570MB. Subsequent runs load from cache.")
        _model = SentenceTransformer(model_name)
        print("Embedding model ready.")
    return _model


class EmbeddingService:
    """
    Wrapper around BGE-M3 that provides simple embed_text / embed_batch methods.

    All vectors are L2-normalized (normalize_embeddings=True), which means
    cosine similarity in Qdrant is equivalent to dot product similarity.
    This gives more accurate search results.
    """

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single string and return a 1024-dimensional float list.

        Args:
            text: Any string in French, Arabic, Darija, or English.

        Returns:
            List of 1024 floats representing the semantic meaning of the text.

        Usage:
            vector = service.embed_text("What are the payment terms?")
            # Use this vector to search Qdrant for similar documents
        """
        model = _get_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple strings in a single forward pass — much faster than
        calling embed_text() in a loop when you have many documents.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 1024-dimensional vectors, one per input text.
            Order is preserved: texts[i] → result[i]

        Usage:
            # When ingesting a batch of document chunks
            chunks = ["clause 1 text", "clause 2 text", "clause 3 text"]
            vectors = service.embed_batch(chunks)
            # Now store each (chunk, vector) pair in Qdrant
        """
        model = _get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]


# ── Quick smoke test — run this file directly to verify the model loads ───────
if __name__ == "__main__":
    print("Testing EmbeddingService...")
    service = EmbeddingService()

    # Test single embedding
    vector = service.embed_text("Test document about AI infrastructure")
    print(f"Single embedding: {len(vector)} dimensions (expected 1024)")
    assert len(vector) == 1024, "Unexpected dimension count"

    # Test batch embedding
    texts = ["French text", "Arabic text", "English text"]
    vectors = service.embed_batch(texts)
    print(f"Batch embedding: {len(vectors)} vectors of {len(vectors[0])} dims each")
    assert len(vectors) == 3
    assert all(len(v) == 1024 for v in vectors)

    print("All tests passed.")
