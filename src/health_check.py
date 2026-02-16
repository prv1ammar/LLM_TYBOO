"""
Health check script to verify all services are running
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
VLLM_URL = "http://localhost:8000"
TEI_URL = "http://localhost:8080"

def check_service(name: str, url: str, endpoint: str = "/health") -> bool:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} is healthy")
            return True
        else:
            print(f"⚠️  {name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name} is not reachable: {e}")
        return False

def check_litellm_models():
    """Check if LiteLLM can see the configured models"""
    try:
        response = requests.get(f"{LITELLM_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ LiteLLM models available: {models}")
            return True
        else:
            print(f"⚠️  Could not fetch LiteLLM models")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ LiteLLM models check failed: {e}")
        return False

def main():
    print("🔍 Checking service health...\n")
    
    results = {
        "vLLM": check_service("vLLM", VLLM_URL),
        "TEI": check_service("TEI", TEI_URL),
        "LiteLLM": check_service("LiteLLM", LITELLM_URL),
    }
    
    print("\n🔍 Checking LiteLLM configuration...\n")
    check_litellm_models()
    
    print("\n" + "="*50)
    if all(results.values()):
        print("✅ All services are healthy!")
    else:
        print("⚠️  Some services are not healthy. Check the logs above.")
        print("\nTo start services, run: docker-compose up -d")

if __name__ == "__main__":
    main()
