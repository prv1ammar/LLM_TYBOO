"""
model_router.py — Smart LLM Router
====================================
PURPOSE:
  Routes every query to the most appropriate model based on complexity.
  This avoids wasting the slow 14B model on simple greetings, while
  ensuring complex tasks always get the best quality answer.

HOW THE TWO MODELS WORK:
  Qwen2.5-14B (port 8000 → "internal-llm-14b")
    - High quality, slower (~3-5 tokens/sec on CPU)
    - Used for: RAG queries, document analysis, contract review,
      code generation, debugging, complex reasoning, anything technical.

  Qwen2.5-3B (port 8001 → "internal-llm-3b")
    - Fast, lightweight (~8-12 tokens/sec on CPU)
    - Used for: greetings, short factual questions, simple Q&A,
      quick summaries, anything under 80 characters.

ROUTING DECISION ORDER (checked top to bottom):
  1. Query matches a COMPLEX pattern  →  always use 14B
  2. Query matches a SIMPLE pattern   →  use 3B
  3. Query length < 80 characters     →  use 3B
  4. Default fallback                 →  use 14B (quality over speed)

HOW MODELS ARE CONNECTED:
  Both models are exposed through LiteLLM proxy (port 4000).
  LiteLLM translates standard OpenAI API calls to llama-cpp-python format.
  This means n8n, LangChain, or any OpenAI-compatible client can call:
    - model: "internal-llm-14b"  →  goes to llm-14b container
    - model: "internal-llm-3b"   →  goes to llm-3b container
    - model: "internal-llm"      →  goes to 14B (default alias)

HOW TO USE THIS MODULE:
  from model_router import get_model_for_chat, get_model_for_rag, get_model

  # Auto-routing based on query (most common usage)
  model, label = get_model_for_chat("Analyse this contract")
  # label will be "14B" — "contract" triggers complex pattern

  # Always get 14B for document retrieval (RAG needs accuracy)
  model = get_model_for_rag()

  # Get default model (14B) without routing logic
  model = get_model()

TO TEST THE ROUTER:
  python model_router.py
  # Runs built-in test suite and shows PASS/FAIL for each case
"""

import os
import re
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()

# LiteLLM proxy — single entry point for all model calls
# Docker internal: http://litellm:4000
# Local dev:       http://localhost:4000
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "sk-tyboo-2025")


# ── Complex patterns → always route to 14B ───────────────────────────────────
# These patterns cover French, Arabic, and English.
# If ANY pattern matches, 14B is used regardless of query length.
_COMPLEX_PATTERNS = [
    # Analytical / reasoning verbs
    r"(analys|compar|expliqu|détail|résume|synthès|évalue|stratég)",   # French
    r"(حلل|قارن|ناقش|اشرح|استنتج|فرق|استراتيج)",                      # Arabic
    r"(analy[sz]|compar|explain|detail|evaluat|strateg|review)",        # English

    # Legal / financial / compliance
    r"(contrat|juridique|légal|fiscal|comptab|financ|budget|audit)",    # French
    r"(قانون|عقد|ضريبة|محاسبة|ميزانية|مالية)",                          # Arabic
    r"(contract|legal|tax|accounting|compliance|invoice|clause)",        # English

    # Code and technical work
    r"(code|script|fonction|class|debug|erreur|exception|api|sql)",     # French
    r"(كود|برمجة|خوارزمية)",                                             # Arabic
    r"(function|class|debug|error|implement|refactor|algorithm|query)", # English

    # Deep reasoning questions
    r"(pourquoi|comment est-ce|quelles sont les raisons)",              # French
    r"(لماذا|كيف يمكن|ما هي الأسباب|ما الفرق)",                        # Arabic
    r"(why|how does|what are the reasons|what is the difference)",      # English

    # Any query longer than 150 characters is assumed complex
    r".{150,}",
]

# ── Simple patterns → route to 3B for fast response ─────────────────────────
# These match short conversational or factual queries.
_SIMPLE_PATTERNS = [
    # Greetings and one-word responses
    r"^(bonjour|salut|merci|ok|oui|non|bonsoir|au revoir)\b",          # French
    r"^(salam|مرحبا|شكرا|صباح|مساء|نعم|لا)\b",                         # Arabic
    r"^(hello|hi|thanks|thank you|yes|no|okay|bye|good morning)\b",    # English

    # Short factual questions (under ~40 chars after the keyword)
    r"^(qu.est.ce que|c.est quoi|c.est qui|qui est|quand|ou)\s.{0,40}", # French
    r"^(ما|من|متى|أين|كم|هل)\s.{0,35}[\?؟]?$",                          # Arabic
    r"^(what is|who is|where is|when is|how many|is it)\s.{0,40}",     # English

    # Simple list or enumeration requests
    r"^(liste|donne.moi|enumere|cite)\s.{0,50}",                        # French
    r"^(قائمة|اذكر|عدد|هات)\s.{0,50}",                                  # Arabic
    r"^(list|give me|show me|name)\s.{0,50}",                           # English
]


def _build_model(model_name: str) -> OpenAIModel:
    """
    Create a PydanticAI model instance that connects to LiteLLM proxy.

    LiteLLM receives the OpenAI-format request and forwards it to the
    correct llama-cpp-python server based on model_name.

    Args:
        model_name: Must match a name defined in litellm_config.yaml.
                    Valid values: "internal-llm-14b", "internal-llm-3b", "internal-llm"
    """
    provider = OpenAIProvider(
        base_url=f"{LITELLM_URL}/v1",
        api_key=LITELLM_KEY,
    )
    return OpenAIModel(model_name, provider=provider)


def route(query: str) -> tuple[OpenAIModel, str]:
    """
    Inspect query and return the appropriate model.

    Args:
        query: The raw user input string.

    Returns:
        Tuple of (OpenAIModel instance, label string)
        Label is "14B" or "3B" — useful for logging.

    Example:
        model, label = route("Analyse this contract")
        print(label)  # "14B"

        model, label = route("Hello!")
        print(label)  # "3B"
    """
    q = query.strip()

    # Complex patterns take priority — check these first
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return _build_model("internal-llm-14b"), "14B"

    # Simple patterns — fast path to 3B
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return _build_model("internal-llm-3b"), "3B"

    # Short queries that didn't match any pattern → 3B
    if len(q) < 80:
        return _build_model("internal-llm-3b"), "3B"

    # Fallback: unclassified queries → 14B (safe default)
    return _build_model("internal-llm-14b"), "14B"


def get_model_for_chat(query: str) -> tuple[OpenAIModel, str]:
    """
    Auto-select model for agent and chat use cases.
    Call this whenever you need a model for a user-driven query.

    Usage:
        model, label = get_model_for_chat(user_input)
        agent = Agent(model, system_prompt="You are a helpful assistant.")
        result = await agent.run(user_input)
    """
    return route(query)


def get_model_for_rag() -> OpenAIModel:
    """
    Always returns the 14B model — RAG requires maximum accuracy.

    RAG answers are grounded in real documents. Using 3B risks
    hallucinations or poor synthesis of retrieved context.
    Always use 14B for any RAG pipeline, regardless of query length.

    Usage:
        model = get_model_for_rag()
        agent = Agent(model, system_prompt="Answer only from context...")
    """
    return _build_model("internal-llm-14b")


def get_model() -> OpenAIModel:
    """
    Returns the default 14B model instance.
    Use this when you don't need routing logic (e.g. scheduled jobs,
    batch processing, or any task where you always want full quality).

    Usage:
        model = get_model()
        agent = Agent(model, system_prompt="...")
    """
    return _build_model("internal-llm-14b")


# ── Self-test — run directly to verify routing logic works correctly ──────────
if __name__ == "__main__":
    # Format: (query, expected_model_label)
    # Run with:  python model_router.py
    tests = [
        ("Bonjour, comment ca va ?",                              "3B"),
        ("Hello how are you",                                     "3B"),
        ("What is the capital of Morocco?",                       "3B"),
        ("Who is the CEO?",                                       "3B"),
        ("merci",                                                  "3B"),
        ("yes",                                                   "3B"),
        ("Give me a list of cities",                              "3B"),
        ("Analyse this contract and find abusive clauses",        "14B"),
        ("Write a Python function to parse CSV and return JSON",  "14B"),
        ("What is the difference between RAG and fine-tuning?",   "14B"),
        ("Compare the offers from suppliers A, B and C",         "14B"),
        ("Explain in detail the data ingestion pipeline",         "14B"),
        ("Why is the API returning a 500 error?",                 "14B"),
        ("Review this SQL query for performance issues",          "14B"),
        ("Debug this Python class and fix the memory leak",       "14B"),
    ]

    print("Model Router — Self-Test")
    print("=" * 58)
    passed = 0
    for query, expected in tests:
        _, chosen = route(query)
        ok = chosen == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] [{chosen:>3}]  {query[:52]}")

    print("=" * 58)
    print(f"  Result: {passed}/{len(tests)} passed")
