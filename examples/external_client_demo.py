"""
Example: External Client Application
This script mimics a completely separate project (e.g., a Data Analysis App)
using the shared LLM backend.
"""
import sys
import os
import time

# Add sdk directory to path (in a real project, you'd copy the sdk file)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdk'))

from llm_client import LLMBackendClient

def run_data_processing_pipeline():
    print("🚀 Starting Data Processing Client...")
    
    # Connect to the central LLM service
    client = LLMBackendClient(
        api_url="http://localhost:8888",
        api_key="your-secure-api-key-here"  # Matches .env default
    )
    
    # 1. Check health
    try:
        health = client.health()
        print(f"✅ Backend is healthy: {health}")
    except Exception as e:
        print(f"❌ Could not connect to backend: {e}")
        return

    # 2. Simulate processing a large batch of customer feedback
    customer_feedback = [
        "The service was excellent and very fast.",
        "I had trouble logging in, please fix it.",
        "Pricing is too high for the features provided.",
        "Great support team, helped me resolove my issue.",
        "The UI is confusing and hard to navigate."
    ]
    
    print(f"\n📨 Submitting batch analysis job for {len(customer_feedback)} items...")
    
    # We use the job system for this
    job_id = client.submit_job(
        job_type="batch_chat",
        params={
            "messages": [
                f"Analyze sentiment and extract key issue: '{msg}'" 
                for msg in customer_feedback
            ]
        }
    )
    
    print(f"⏳ Job submitted (ID: {job_id}). Waiting for results...")
    
    # Poll for completion
    try:
        result = client.get_job_result(job_id, wait=True, timeout=60)
        print("\n📊 Analysis Results:")
        for original, analysis in zip(customer_feedback, result['responses']):
            print(f"- Input: {original[:30]}...")
            print(f"  Analysis: {analysis[:100]}...\n")
            
    except Exception as e:
        print(f"❌ Job failed or timed out: {e}")

    # 3. Create a project-specific knowledge base
    print("📚 Updating Project Knowledge Base...")
    
    docs = [
        {"text": "Error 500 means internal server error.", "metadata": {"Topic": "Errors"}},
        {"text": "To reset password, go to settings page.", "metadata": {"Topic": "Auth"}}
    ]
    
    # Ingest into a specific collection named 'app_support'
    client.rag_ingest(docs, collection="app_support")
    
    # Query that specific collection
    print("❓ Querying knowledge base...")
    answer = client.rag_query(
        "How do I fix a 500 error?", 
        collection="app_support"
    )
    
    print(f"💡 Answer: {answer['answer']['answer']}")

if __name__ == "__main__":
    run_data_processing_pipeline()
