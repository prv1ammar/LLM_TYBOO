"""
health_check.py — Service Health Checker
==========================================
PURPOSE:
  Verifies that all Docker services are running and responding correctly
  before you start using the platform.

  Run this after "docker compose up -d" to confirm everything is healthy.

WHAT IT CHECKS:
  1. llama-cpp LLM (14B) — port 8000 — model server responding
  2. llama-cpp LLM (3B)  — port 8001 — model server responding
  3. sentence-transformers — Python package installed correctly
  4. Qdrant               — port 6333 — vector database responding
  5. LiteLLM proxy        — port 4000 — proxy layer responding
  6. LiteLLM inference    — sends a real test prompt to verify end-to-end
  7. FastAPI              — port 8888 — API responding
  8. n8n                  — port 5678 — automation platform responding

WHAT THE OUTPUT LOOKS LIKE:
  [OK] llama-cpp LLM 14B — http://localhost:8000
  [OK] llama-cpp LLM 3B  — http://localhost:8001
  [OK] sentence-transformers 3.x.x
  [OK] Qdrant            — http://localhost:6333
  [OK] LiteLLM proxy     — http://localhost:4000
  [OK] LiteLLM inference — model responding
  [OK] FastAPI API        — http://localhost:8888
  [OK] n8n               — http://localhost:5678

  n8n / LangChain credentials:
    Base URL : http://YOUR_SERVER_IP:4000/v1
    API Key  : sk-tyboo-2025
    Model    : internal-llm

HOW TO RUN:
  cd src
  python health_check.py

  Or from project root:
  python src/health_check.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Service URLs — these match the Docker Compose service names and ports
LLM_14B_URL = os.getenv("LLM_14B_URL", "http://localhost:8000")
LLM_3B_URL  = os.getenv("LLM_3B_URL",  "http://localhost:8001")
QDRANT_URL  = os.getenv("QDRANT_URL",  "http://localhost:6333")
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-tyboo-2025")
API_URL     = "http://localhost:8888"
N8N_URL     = "http://localhost:5678"


def check_http(label: str, url: str, path: str = "/health", headers: dict = None) -> bool:
    """
    Send a GET request and check that the service responds with 200 or 401.

    401 counts as "healthy" because it means the service is running
    but requires authentication — LiteLLM returns 401 on /health
    when called without a key.

    Args:
        label:   Display name shown in the output
        url:     Base URL of the service
        path:    Health check path (default /health)
        headers: Optional request headers

    Returns:
        True if service is reachable and responding, False otherwise
    """
    try:
        r = requests.get(f"{url}{path}", timeout=10, headers=headers or {})
        if r.status_code in (200, 401):
            print(f"  [OK] {label:<35} {url}")
            return True
        print(f"  [WARN] {label:<33} {url}  — HTTP {r.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] {label:<33} {url}  — Connection refused (service not running?)")
        return False
    except requests.exceptions.Timeout:
        print(f"  [FAIL] {label:<33} {url}  — Timeout (service overloaded or starting?)")
        return False
    except Exception as e:
        print(f"  [FAIL] {label:<33} {url}  — {str(e)}")
        return False


def check_embeddings() -> bool:
    """
    Verify that sentence-transformers is installed and importable.

    This doesn't load the model — just confirms the package is present.
    The model is loaded lazily on first use (see embeddings.py).
    """
    try:
        import sentence_transformers
        print(f"  [OK] {'sentence-transformers':<35} v{sentence_transformers.__version__}")
        return True
    except ImportError:
        print("  [FAIL] sentence-transformers                 — not installed")
        print("         Fix: pip install sentence-transformers")
        return False


def check_litellm_inference() -> bool:
    """
    Send a real test prompt through LiteLLM to verify end-to-end model inference.

    This is the most important check — it confirms that:
      1. LiteLLM proxy is running
      2. LiteLLM can reach the llama-cpp container
      3. The model is loaded and generating responses

    Uses max_tokens=5 to keep the test fast (we just need to confirm it works).
    This check may take 10-30 seconds on first call if the model is warming up.
    """
    try:
        r = requests.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json={
                "model": "internal-llm-3b",  # Use 3B for the test — faster response
                "messages": [{"role": "user", "content": "Say: OK"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            timeout=60,  # Model may need time to warm up
        )
        if r.status_code == 200:
            print(f"  [OK] {'LiteLLM inference (3B test)':<35} model responding")
            return True
        print(f"  [FAIL] {'LiteLLM inference':<33} HTTP {r.status_code}: {r.text[:100]}")
        return False
    except requests.exceptions.Timeout:
        print("  [FAIL] LiteLLM inference                     — Timeout (model still loading?)")
        print("         Wait 2-3 minutes and run health_check.py again")
        return False
    except Exception as e:
        print(f"  [FAIL] LiteLLM inference                     — {str(e)}")
        return False


def main():
    print("LLM_TYBOO Health Check — Dual Model CPU Edition")
    print("=" * 58)

    # Run all checks and collect results
    results = {
        "llama-cpp LLM 14B":    check_http("llama-cpp LLM 14B", LLM_14B_URL),
        "llama-cpp LLM 3B":     check_http("llama-cpp LLM 3B",  LLM_3B_URL),
        "sentence-transformers": check_embeddings(),
        "Qdrant":               check_http("Qdrant", QDRANT_URL, "/healthz"),
        "LiteLLM proxy":        check_http("LiteLLM proxy", LITELLM_URL),
        "LiteLLM inference":    check_litellm_inference(),
        "FastAPI":              check_http("FastAPI API", API_URL),
        "n8n":                  check_http("n8n", N8N_URL, "/healthz"),
    }

    print("=" * 58)

    all_ok = all(results.values())
    if all_ok:
        print("  All services are healthy — platform is ready!")
        print()
        print("  n8n / LangChain / LiteLLM node credentials:")
        print(f"    Base URL : http://YOUR_SERVER_IP:4000/v1")
        print(f"    API Key  : {LITELLM_KEY}")
        print(f"    Model    : internal-llm        (14B — default)")
        print(f"    Model    : internal-llm-14b    (14B — explicit)")
        print(f"    Model    : internal-llm-3b     (3B  — fast)")
        print()
        print("  Service URLs:")
        print("    n8n Dashboard    : http://YOUR_SERVER_IP:5678")
        print("    API Swagger      : http://YOUR_SERVER_IP:8888/docs")
        print("    Streamlit        : http://YOUR_SERVER_IP:8501")
        print("    Qdrant Dashboard : http://YOUR_SERVER_IP:6333/dashboard")
        print("    Grafana          : http://YOUR_SERVER_IP:3000")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  Issues detected in: {', '.join(failed)}")
        print()
        print("  Troubleshooting:")
        print("    docker compose ps           — check container status")
        print("    docker compose logs <name>  — view service logs")
        print("    docker compose up -d        — start stopped services")


if __name__ == "__main__":
    main()
