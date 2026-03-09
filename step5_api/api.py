"""
RAVEN - Step 5: FastAPI Orchestration API
==========================================
Point d'entrée principal qui orchestre tous les steps:

  POST /chat      → pipeline complet: guardrails → intent → LLM → summary
  GET  /session   → historique + profil utilisateur
  GET  /summary   → résumé de session
  POST /feedback  → feedback sur les réponses
  GET  /health    → health check

Flow complet:
  User Message
    → [Step 3] GuardrailPipeline (PII scan + Redis save + JSONL log)
    → [Step 2] IntentClassifier  (intent + tags)
    → [Step 1 models] Qwen LLM   (réponse finale, anti-répétition)
    → [Step 4] Summarizer        (résumé session en background)
    → Response to user
"""

import uuid
import time
import json
import asyncio
from typing import Optional, List
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise RuntimeError("pip install fastapi uvicorn pydantic")

# Import RAVEN steps
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from step2_intent_classifier.classifier import IntentClassifier, IntentResult
from step3_guardrails.guardrails import GuardrailPipeline, ScanResult
from step4_summarizer.summarizer import ConversationSummarizer, SummaryStore


# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════

class Config:
    REDIS_HOST     = "localhost"
    REDIS_PORT     = 6379
    LOG_DIR        = "logs"
    INTENT_MODEL   = "models/qwen_intent_finetuned"
    LLM_MODEL      = "Qwen/Qwen2.5-0.5B-Instruct"   # main chat model
    LLM_MODEL_BIG  = "Qwen/Qwen2.5-1.5B-Instruct"   # fallback / summarizer
    MAX_HISTORY    = 20  # max turns sent to LLM
    REPETITION_THRESHOLD = 0.6  # si score > seuil → forcer variation


# ══════════════════════════════════════════════════════
#  API SCHEMAS
# ══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    user_id: str = Field(..., min_length=1, max_length=64)
    session_id: Optional[str] = None
    lang_hint: Optional[str] = None  # "arabic"|"darija"|"french"|"english"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    tags: List[str]
    lang: str
    confidence: float
    pii_detected: bool
    pii_types: List[str]
    was_blocked: bool
    risk_level: str
    processing_time_ms: float


class SessionInfo(BaseModel):
    user_id: str
    session_id: str
    lang: str
    message_count: int
    intents_seen: List[str]
    tags_seen: List[str]
    pii_types_seen: List[str]
    flagged_count: int
    history_preview: List[dict]


class SummaryResponse(BaseModel):
    session_id: str
    summary_text: str
    primary_intent: str
    all_intents: List[str]
    all_tags: List[str]
    risk_level: str
    recommended_actions: List[str]
    needs_human_takeover: bool
    escalation_reason: str


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    redis: bool
    intent_model: bool
    llm_model: bool
    uptime_seconds: float
    version: str = "1.0.0"


# ══════════════════════════════════════════════════════
#  LLM WRAPPER (anti-répétition)
# ══════════════════════════════════════════════════════

class LLMEngine:
    """
    Wraps Qwen for chat with built-in repetition detection.
    """
    SYSTEM_PROMPT = """You are RAVEN, a helpful multilingual assistant.
Languages supported: Arabic (العربية), Moroccan Darija (الدارجة), French, English.
Always respond in the same language as the user.

CRITICAL RULES:
1. NEVER repeat information you already provided in this conversation
2. Each response must add NEW value — new info, new angle, or ask a clarifying question
3. If you have nothing new to add, say so honestly and ask what else the user needs
4. Be concise and helpful
5. NEVER expose personal data — use masked tokens like [RIB_MASQUÉ] if needed"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self._model.eval()

    def _repetition_score(self, new_response: str, history: List[dict]) -> float:
        """Simple token overlap score between new response and recent assistant turns."""
        past = " ".join(t["content"] for t in history if t.get("role") == "assistant")
        if not past:
            return 0.0
        new_words = set(new_response.lower().split())
        past_words = set(past.lower().split())
        if not new_words:
            return 1.0
        overlap = len(new_words & past_words) / len(new_words)
        return overlap

    def generate(self, history: List[dict], user_message: str,
                 intent: str = "", tags: List[str] = None) -> str:
        import torch

        if self._model is None:
            self._load()

        tags = tags or []

        # Build context — inject intent hint
        system = self.SYSTEM_PROMPT
        if intent:
            system += f"\n\nDetected user intent: {intent}. Tags: {', '.join(tags)}."

        messages = [{"role": "system", "content": system}]
        # Add recent history (last MAX_HISTORY turns)
        for turn in history[-Config.MAX_HISTORY:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,   # built-in anti-repetition
            )

        response = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Check repetition score
        rep_score = self._repetition_score(response, history)
        if rep_score > Config.REPETITION_THRESHOLD:
            # Force a different generation with higher temperature
            with torch.no_grad():
                out2 = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.2,
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.5,
                )
            response = self._tokenizer.decode(
                out2[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

        return response.strip()


# ══════════════════════════════════════════════════════
#  APP INIT
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="RAVEN API",
    description="Multilingual LLM pipeline with guardrails, intent classification & summarization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded components
_guardrail: Optional[GuardrailPipeline] = None
_classifier: Optional[IntentClassifier] = None
_llm: Optional[LLMEngine] = None
_summarizer: Optional[ConversationSummarizer] = None
_summary_store: Optional[SummaryStore] = None
_start_time = time.time()


def get_guardrail() -> GuardrailPipeline:
    global _guardrail
    if _guardrail is None:
        _guardrail = GuardrailPipeline(
            redis_host=Config.REDIS_HOST,
            redis_port=Config.REDIS_PORT,
            log_dir=Config.LOG_DIR,
        )
    return _guardrail


def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier(Config.INTENT_MODEL)
    return _classifier


def get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        _llm = LLMEngine(Config.LLM_MODEL)
    return _llm


def get_summarizer() -> ConversationSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = ConversationSummarizer(
            model_path=Config.LLM_MODEL_BIG,
            use_rule_fallback=True,
        )
    return _summarizer


def get_summary_store() -> SummaryStore:
    global _summary_store
    if _summary_store is None:
        _summary_store = SummaryStore(log_dir=Config.LOG_DIR)
    return _summary_store


# ══════════════════════════════════════════════════════
#  BACKGROUND: ASYNC SUMMARIZER
# ══════════════════════════════════════════════════════

async def _background_summarize(session_id: str, user_id: str, lang: str,
                                  history: list, intents: list, tags: list, pii_types: list):
    """Run summarization in background after each response."""
    try:
        summarizer = get_summarizer()
        store = get_summary_store()
        summary = summarizer.summarize(
            session_id=session_id, user_id=user_id, lang=lang,
            history=history, intents=intents, tags=tags, pii_types=pii_types,
        )
        store.save(summary)
    except Exception as e:
        print(f"[Step5] ⚠️ Background summarization failed: {e}")


# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    t0 = time.time()

    # Ensure session_id
    session_id = req.session_id or str(uuid.uuid4())
    guardrail = get_guardrail()

    # ── Step 3: Guardrails (first pass on user message) ──
    masked_msg, scan = guardrail.process(
        user_id=req.user_id,
        session_id=session_id,
        message=req.message,
        role="user",
        lang=req.lang_hint or "unknown",
    )

    if scan.is_blocked:
        return ChatResponse(
            session_id=session_id,
            response=f"⛔ {scan.block_reason}",
            intent="blocked",
            tags=["blocked"],
            lang=req.lang_hint or "unknown",
            confidence=1.0,
            pii_detected=False,
            pii_types=[],
            was_blocked=True,
            risk_level="critical",
            processing_time_ms=(time.time() - t0) * 1000,
        )

    # ── Step 2: Intent Classification ────────────────────
    classifier = get_classifier()
    intent_result: IntentResult = classifier.classify(masked_msg)
    lang = intent_result.lang if intent_result.lang != "unknown" else (req.lang_hint or "french")

    # Update guardrail log with intent info
    guardrail.logger.log(
        user_id=req.user_id, session_id=session_id, role="intent",
        original_text="", masked_text="",
        scan=scan, intent=intent_result.intent, tags=intent_result.tags, lang=lang,
    )

    # ── Step 1 (Models): LLM Generation ──────────────────
    history = guardrail.redis.get_history(session_id)
    llm = get_llm()
    response_text = llm.generate(
        history=history,
        user_message=masked_msg,
        intent=intent_result.intent,
        tags=intent_result.tags,
    )

    # ── Step 3 (again): Scan LLM response for PII leak ───
    masked_response, response_scan = guardrail.process(
        user_id=req.user_id,
        session_id=session_id,
        message=response_text,
        role="assistant",
        intent=intent_result.intent,
        tags=intent_result.tags,
        lang=lang,
    )

    # Risk level
    risk = "low"
    if intent_result.intent == "harmful" or response_scan.is_blocked:
        risk = "critical"
    elif scan.has_pii or "requires_human" in intent_result.tags or intent_result.intent == "sensitive":
        risk = "high"
    elif scan.has_pii:
        risk = "medium"

    # ── Step 4: Background Summarization ─────────────────
    updated_history = guardrail.redis.get_history(session_id)
    profile = guardrail.redis.get_user_profile(req.user_id)
    background_tasks.add_task(
        _background_summarize,
        session_id=session_id,
        user_id=req.user_id,
        lang=lang,
        history=updated_history,
        intents=profile.intents_seen if profile else [intent_result.intent],
        tags=profile.tags_seen if profile else intent_result.tags,
        pii_types=list(profile.pii_detected.keys()) if profile else scan.pii_types,
    )

    return ChatResponse(
        session_id=session_id,
        response=masked_response,
        intent=intent_result.intent,
        tags=intent_result.tags,
        lang=lang,
        confidence=intent_result.confidence,
        pii_detected=scan.has_pii,
        pii_types=scan.pii_types,
        was_blocked=scan.is_blocked,
        risk_level=risk,
        processing_time_ms=(time.time() - t0) * 1000,
    )


@app.get("/session/{user_id}/{session_id}", response_model=SessionInfo)
async def get_session(user_id: str, session_id: str):
    guardrail = get_guardrail()
    profile = guardrail.redis.get_user_profile(user_id)
    history = guardrail.redis.get_history(session_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfo(
        user_id=user_id,
        session_id=session_id,
        lang=profile.lang,
        message_count=profile.message_count,
        intents_seen=profile.intents_seen,
        tags_seen=profile.tags_seen,
        pii_types_seen=list(profile.pii_detected.keys()),
        flagged_count=profile.flagged_count,
        history_preview=history[-5:],
    )


@app.get("/summary/{session_id}", response_model=SummaryResponse)
async def get_summary(session_id: str):
    store = get_summary_store()
    summary = store.get(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found for this session")
    return SummaryResponse(
        session_id=summary.session_id,
        summary_text=summary.summary_text,
        primary_intent=summary.primary_intent,
        all_intents=summary.all_intents,
        all_tags=summary.all_tags,
        risk_level=summary.risk_level,
        recommended_actions=summary.recommended_actions,
        needs_human_takeover=summary.needs_human_takeover,
        escalation_reason=summary.escalation_reason,
    )


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    log_path = Path(Config.LOG_DIR) / "feedback.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    entry = {
        "ts": time.time(),
        "session_id": req.session_id,
        "message_index": req.message_index,
        "rating": req.rating,
        "comment": req.comment,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/health", response_model=HealthResponse)
async def health():
    redis_ok = False
    try:
        guardrail = get_guardrail()
        guardrail.redis._ensure_connected()
        redis_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        redis=redis_ok,
        intent_model=_classifier is not None,
        llm_model=_llm is not None,
        uptime_seconds=time.time() - _start_time,
    )


# ══════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
