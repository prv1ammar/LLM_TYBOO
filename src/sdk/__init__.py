"""
Tython Python SDK
==================
Official Python client for the Tython AI Platform.

Quick start:
    from tython import TythonClient

    client = TythonClient(
        api_url="http://YOUR_SERVER_IP:8888",
        api_key="your-api-key"
    )

    # Chat
    answer = client.chat("Hello, what can you do?")

    # Embeddings
    vectors = client.embed(["text one", "text two"])

    # RAG
    result = client.rag_query("What is the refund policy?", collection="knowledge_base")
    print(result["answer"])
"""

from .tython_client import TythonClient, embed, chat, rag_query

__version__ = "1.0.0"
__author__ = "Tython"
__all__ = ["TythonClient", "embed", "chat", "rag_query"]
