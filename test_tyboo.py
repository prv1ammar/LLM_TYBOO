from tython import TythonClient
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
# Using the production IP from your environment
SERVER_IP = os.getenv("SERVER_IP", "135.125.4.184")
API_KEY = os.getenv("API_KEY", "92129f24-12f0-478a-8c98-5b15070235e6")

print(f"Connecting to Tython AI Platform at {SERVER_IP}:8888...")

client = TythonClient(
    api_url=f"http://{SERVER_IP}:8888",
    api_key=API_KEY
)

try:
    print("\n1. Testing Chat (Auto-Routing to 3B or 14B)...")
    answer = client.chat("What are the key obligations in this contract?")
    print(f"   > Answer: {answer}")

    print("\n2. Testing RAG (Knowledge Base Search)...")
    # Note: 'enterprise_kb' is the default collection in the unified API
    result = client.rag_query("What is the refund policy?", collection="enterprise_kb")
    print(f"   > Answer from KB: {result.get('answer', 'No answer found')}")

    print("\n3. Testing Embeddings (BGE-M3)...")
    vectors = client.embed(["text 1", "text 2"])
    print(f"   > Generated {len(vectors)} vectors of dimension {len(vectors[0])} each.")

    print("\n✅ ALL TESTS SUCCESSFUL (assuming server is reachable)")

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    print("\nDIAGNOSTIC HINT:")
    print("If you get [WinError 10061], it means the remote server is down or port 8888 is blocked.")
    print("If you get a 401, check that the API_KEY matches in both .env and this script.")
