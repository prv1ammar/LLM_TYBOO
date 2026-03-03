"""
LLM_TYBOO — Professional Connection & Capability Test
======================================================
This script verifies the connection to the Tython AI Platform
and tests its core capabilities: Chat, RAG, and Embeddings.

Usage:
    python test_tyboo_sdk.py
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TybooTester")

# Add SDK to path
sdk_path = os.path.join(os.getcwd(), "src", "sdk")
if os.path.exists(sdk_path):
    sys.path.append(sdk_path)
else:
    logger.error(f"SDK path not found at {sdk_path}. Please run from the project root.")
    sys.exit(1)

try:
    from tython_client import TythonClient
except ImportError:
    logger.error("Could not import TythonClient from sdk folder.")
    sys.exit(1)

def print_result(title: str, data: Any):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(data)
    print(f"{'='*60}\n")

def run_diagnostic():
    # Load settings from .env or use defaults
    # For testing, we use the values provided in the prompt
    API_URL = "http://135.125.4.184:8888"
    API_KEY = "92129f24-12f0-478a-8c98-5b15070235e6"

    logger.info(f"Initializing TythonClient with URL: {API_URL}")
    client = TythonClient(api_url=API_URL, api_key=API_KEY)

    # 1. Health Check
    try:
        logger.info("Step 1: Checking System Health...")
        health = client.health()
        print_result("SYSTEM HEALTH", health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.warning("Aborting further tests due to connection issues.")
        return

    # 2. Chat Test
    try:
        logger.info("Step 2: Testing Chat Completion (Auto-Routing)...")
        msg = "What are the key obligations in a standard service level agreement?"
        answer = client.chat(msg)
        print_result("CHAT RESPONSE", {"question": msg, "answer": answer})
    except Exception as e:
        logger.error(f"Chat test failed: {e}")

    # 3. RAG Test
    try:
        logger.info("Step 3: Testing RAG Query (Knowledge Base Search)...")
        question = "What is the refund policy?"
        # Note: Ensure the collection 'enterprise_kb' exists or use 'default'
        result = client.rag_query(question, collection="enterprise_kb")
        print_result("RAG QUERY RESULT", result)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")

    # 4. Embeddings Test
    try:
        logger.info("Step 4: Testing BGE-M3 Embeddings (1024D)...")
        texts = ["Artificial Intelligence", "Machine Learning"]
        vectors = client.embed(texts)
        logger.info(f"Successfully generated {len(vectors)} embeddings.")
        print_result("EMBEDDINGS (First 10 dimensions)", {
            "text": texts[0],
            "dimensions": len(vectors[0]),
            "preview": vectors[0][:10]
        })
    except Exception as e:
        logger.error(f"Embeddings test failed: {e}")

if __name__ == "__main__":
    run_diagnostic()
