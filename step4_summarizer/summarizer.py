"""
RAVEN - Step 4: Conversation Summarizer
=========================================
Génère un résumé structuré de chaque conversation incluant:
  - Résumé textuel (multilingue)
  - Intentions détectées tout au long de la conversation
  - Tags agrégés
  - Niveau de risque
  - Actions recommandées

Input  : conversation history (de Redis Step 3)
Output : ConversationSummary (sauvé dans Redis + JSONL)
"""

import json
import time
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ══════════════════════════════════════════════════════

@dataclass
class ConversationSummary:
    session_id: str
    user_id: str
    lang: str
    created_at: float = field(default_factory=time.time)

    # Core summary
    summary_text: str = ""              # résumé en langage naturel
    summary_lang: str = "french"        # langue du résumé

    # Intent & Tags
    primary_intent: str = ""           # intent le plus fréquent
    all_intents: List[str] = field(default_factory=list)
    all_tags: List[str] = field(default_factory=list)

    # Stats
    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    avg_user_msg_len: float = 0.0

    # Risk
    risk_level: str = "low"           # low | medium | high | critical
    pii_types_detected: List[str] = field(default_factory=list)
    was_blocked: bool = False

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    needs_human_takeover: bool = False
    escalation_reason: str = ""


# ══════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════

SUMMARIZER_SYSTEM = """You are RAVEN Conversation Summarizer.
Analyze the conversation history and respond ONLY with valid JSON:

{
  "summary_text": "<3-5 sentence summary of what happened in the conversation>",
  "primary_intent": "<main intent: question_info|complaint|transaction|support|off_topic|emergency>",
  "all_intents": ["<list of all detected intents>"],
  "all_tags": ["<aggregated tags from conversation>"],
  "risk_level": "<low|medium|high|critical>",
  "recommended_actions": ["<action1>", "<action2>"],
  "needs_human_takeover": <true|false>,
  "escalation_reason": "<reason if needs_human_takeover is true, else empty string>"
}

Risk level rules:
- critical: emergency intent OR fraud_signal tag OR blocked messages
- high: complaint + frustrated + requires_human
- medium: pii_risk detected OR multiple intents
- low: normal information request

Respond ONLY with JSON. No preamble, no explanation."""


# ══════════════════════════════════════════════════════
#  SUMMARIZER CLASS
# ══════════════════════════════════════════════════════

class ConversationSummarizer:
    """
    Uses fine-tuned Qwen (or base Qwen) to summarize conversations.
    Falls back to rule-based summarizer if model unavailable.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 use_rule_fallback: bool = True):
        self.model_path = model_path
        self.use_rule_fallback = use_rule_fallback
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Step4] Loading summarizer: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        print("[Step4] ✅ Summarizer ready")

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for turn in history:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")[:300]  # truncate long turns
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _rule_based_summary(self, history: List[Dict],
                             intents: List[str], tags: List[str],
                             pii_types: List[str]) -> Dict:
        """Fallback: generate summary using rules when model unavailable."""
        n_turns = len(history)
        user_msgs = [t for t in history if t.get("role") == "user"]
        has_sensitive = "sensitive" in intents
        has_harmful = "harmful" in intents
        has_pii = len(pii_types) > 0

        # Risk
        if has_harmful or "security_risk" in tags:
            risk = "critical"
        elif has_sensitive and "requires_human" in tags:
            risk = "high"
        elif has_pii or len(set(intents)) > 2:
            risk = "medium"
        else:
            risk = "low"

        # Primary intent
        from collections import Counter
        primary = Counter(intents).most_common(1)[0][0] if intents else "info_seeking"

        # Recommended actions
        actions = []
        if has_harmful:
            actions.append("Escalade immédiate vers équipe sécurité")
            actions.append("Bloquer le compte si nécessaire")
        if has_sensitive:
            actions.append("Transmettre pour révision humaine")
        if has_pii:
            actions.append("Vérifier conformité RGPD / données masquées")
        if not actions:
            actions.append("Continuer le suivi standard")

        summary_text = (
            f"Conversation de {n_turns} échanges. "
            f"Intention principale: {primary}. "
            f"{'⚠️ Données sensibles détectées. ' if has_pii else ''}"
            f"{'🚨 Situation d urgence signalée. ' if has_emergency else ''}"
            f"Niveau de risque: {risk}."
        )

        return {
            "summary_text": summary_text,
            "primary_intent": primary,
            "all_intents": list(set(intents)),
            "all_tags": list(set(tags)),
            "risk_level": risk,
            "recommended_actions": actions,
            "needs_human_takeover": risk in ("high", "critical"),
            "escalation_reason": "Situation critique détectée" if risk == "critical" else "",
        }

    def summarize(self, session_id: str, user_id: str, lang: str,
                  history: List[Dict], intents: List[str] = None,
                  tags: List[str] = None, pii_types: List[str] = None) -> ConversationSummary:

        intents = intents or []
        tags = tags or []
        pii_types = pii_types or []

        summary = ConversationSummary(
            session_id=session_id,
            user_id=user_id,
            lang=lang,
            total_turns=len(history),
            user_turns=sum(1 for t in history if t.get("role") == "user"),
            assistant_turns=sum(1 for t in history if t.get("role") == "assistant"),
            pii_types_detected=pii_types,
        )

        # Avg user message length
        user_msgs = [t.get("content", "") for t in history if t.get("role") == "user"]
        if user_msgs:
            summary.avg_user_msg_len = sum(len(m) for m in user_msgs) / len(user_msgs)

        # Try LLM summarization
        llm_result = None
        if self.model_path and not self.use_rule_fallback:
            try:
                if self._model is None:
                    self._load()

                conv_text = self._format_history(history)
                prompt = f"Conversation history:\n{conv_text}\n\nIntents detected: {intents}\nTags: {tags}"

                messages = [
                    {"role": "system", "content": SUMMARIZER_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs, max_new_tokens=512, temperature=0.1, do_sample=False
                    )
                raw = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                llm_result = json.loads(raw.strip())
            except Exception as e:
                print(f"[Step4] ⚠️ LLM failed, using rule-based: {e}")

        # Rule-based fallback
        result = llm_result or self._rule_based_summary(history, intents, tags, pii_types)

        # Fill summary
        summary.summary_text = result.get("summary_text", "")
        summary.primary_intent = result.get("primary_intent", "question_info")
        summary.all_intents = result.get("all_intents", intents)
        summary.all_tags = result.get("all_tags", tags)
        summary.risk_level = result.get("risk_level", "low")
        summary.recommended_actions = result.get("recommended_actions", [])
        summary.needs_human_takeover = result.get("needs_human_takeover", False)
        summary.escalation_reason = result.get("escalation_reason", "")

        return summary


# ══════════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════════

class SummaryStore:
    """Save summaries to Redis + JSONL."""

    KEY_PREFIX = "raven:summary:"
    TTL = 7 * 86_400  # 7 jours

    def __init__(self, redis_client=None, log_dir: str = "logs"):
        self._redis = redis_client
        self.log_path = Path(log_dir) / "summaries.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)

    def save(self, summary: ConversationSummary):
        data = asdict(summary)

        # Redis
        if self._redis:
            key = f"{self.KEY_PREFIX}{summary.session_id}"
            self._redis.setex(key, self.TTL, json.dumps(data))

        # JSONL
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"[Step4] ✅ Summary saved — session={summary.session_id} risk={summary.risk_level}")

    def get(self, session_id: str) -> Optional[ConversationSummary]:
        if not self._redis:
            return None
        key = f"{self.KEY_PREFIX}{session_id}"
        raw = self._redis.get(key)
        if raw:
            return ConversationSummary(**json.loads(raw))
        return None


# ══════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_history = [
        {"role": "user",      "content": "kifash nsib virus ?"},
        {"role": "assistant", "content": "ma nqderch nsa3dek f hadchi"},
        {"role": "user",      "content": "3tini ghir chi exemple"},
        {"role": "assistant", "content": "ana hna ghir bash njaweb 3la l'as2ila l3ama"},
        {"role": "user",      "content": "bghit n'annuler compte [IDENTIFIANT_MASQUÉ] dyali"},
        {"role": "assistant", "content": "wakha, ghadi nbloquih lik daba"},
    ]

    summarizer = ConversationSummarizer(use_rule_fallback=True)
    result = summarizer.summarize(
        session_id="sess_demo_001",
        user_id="user_42",
        lang="darija",
        history=demo_history,
        intents=["harmful", "casual_chat", "sensitive"],
        tags=["security_risk", "frustrated", "requires_human", "pii_risk"],
        pii_types=["id"],
    )

    print("[Step4] 📋 Summary Result:")
    print(f"  Summary : {result.summary_text}")
    print(f"  Intent  : {result.primary_intent}")
    print(f"  Tags    : {result.all_tags}")
    print(f"  Risk    : {result.risk_level}")
    print(f"  Actions : {result.recommended_actions}")
    print(f"  Human?  : {result.needs_human_takeover}")
