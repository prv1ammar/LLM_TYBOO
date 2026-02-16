import sys
import os

# Ensure sdk is reachable
sys.path.append(os.path.join(os.getcwd(), "sdk"))

from llm_client import LLMBackendClient

# Use the key from your .env
API_KEY = "8f7a9c2e-1b4d-4e91-a9f1-7c3e5b2d1a90"

def test():
    print("🚀 Connecting with API Key:", API_KEY)
    
    client = LLMBackendClient(
        api_url="http://localhost:8888",
        api_key=API_KEY
    )
    
    try:
        # 1. Health
        print("\nChecking health...")
        print(client.health())
        
        # 2. Embeddings
        print("\nGenerating embedding...")
        emb = client.embed_single("Test connection")
        print(f"Success! Vector length: {len(emb)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test()
