import sys
import os

# Add the SDK directory to the path so we can import 'tython'
# Note: Since the folder is named 'sdk', we either rename it or use it as a module
sdk_path = os.path.join(os.getcwd(), "src", "sdk")
sys.path.append(sdk_path)

try:
    # If using 'from tython import TythonClient', the package name must match the folder
    # or the folder must have an __init__.py and be in a path as 'tython'
    from tython_client import TythonClient
    print("SUCCESS: SDK import successful.")
except ImportError as e:
    print(f"ERROR: Could not import TythonClient: {e}")
    sys.exit(1)

# Configuration from updated .env
API_URL = "http://135.125.4.184:8888"
API_KEY = "92129f24-12f0-478a-8c98-5b15070235e6"

print(f"Testing connection to {API_URL}...")

client = TythonClient(
    api_url=API_URL,
    api_key=API_KEY
)

try:
    print("\n--- Testing Health Check ---")
    health = client.health()
    print(f"Health Response: {health}")

    print("\n--- Testing Chat ---")
    answer = client.chat("What are the key obligations in this contract?")
    print(f"Chat Answer: {answer}")

    print("\n--- Testing RAG Query ---")
    # Using 'default' or a known collection if possible
    result = client.rag_query("What is the refund policy?", collection="knowledge_base")
    print(f"RAG Result: {result.get('answer', 'No answer field in response')}")

    print("\n--- Testing Embeddings ---")
    vectors = client.embed(["text 1", "text 2"])
    print(f"Embeddings: Generated {len(vectors)} vectors.")

except Exception as e:
    print(f"\nERROR during execution: {e}")
