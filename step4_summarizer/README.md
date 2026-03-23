# Step 4 — Conversation Summarizer

**Location:** `step4_summarizer/`  
**Where to run:** ✅ Runs automatically inside the API (VM)  
**Goal:** Summarize conversations and generate session risk assessments

---

## What it does

After each conversation exchange, the summarizer:

1. Reads the full conversation history from Redis
2. Generates a concise summary of the session
3. Identifies the dominant intent of the session
4. Calculates a **risk level**: `low`, `medium`, or `high`
5. Suggests **recommended actions** based on what was discussed
6. Detects if PII was shared during the conversation
7. Stores the summary back in Redis for future reference

---

## Risk levels explained

| Risk Level | Triggered by | Action |
|------------|-------------|--------|
| `low` | `info_seeking`, `creative`, `casual_chat` | Continue normally |
| `medium` | `sensitive` | Log for review |
| `high` | `harmful` | Block + alert |

---

## How to test it manually

```bash
cd /path/to/project/root
venv\Scripts\activate
python -m step4_summarizer.summarizer
```

### Expected output

```json
{
  "session_id": "demo-session",
  "user_id": "user-123",
  "lang": "en",
  "created_at": 1710000000.0,
  "summary_text": "User asked about Python programming and scientific concepts.",
  "summary_lang": "french",
  "primary_intent": "info_seeking",
  "all_intents": ["info_seeking"],
  "all_tags": ["science", "technology"],
  "total_turns": 4,
  "user_turns": 2,
  "assistant_turns": 2,
  "avg_user_msg_len": 45.5,
  "risk_level": "low",
  "pii_types_detected": [],
  "was_blocked": false,
  "recommended_actions": [
    "Continue helping the user"
  ],
  "needs_human_takeover": false,
  "escalation_reason": ""
}
```

---

## How it works in the API

```
User sends message
    ↓
[Guardrails]     ← Step 3
    ↓
[Intent Classifier]  ← Step 2
    ↓
[LLM responds]   ← Step 5
    ↓
[Summarizer updates session in Redis]  ← Step 4 (async)
    ↓
Response sent to User
```

---

## Get a summary via the API

```bash
curl http://192.168.0.184:8000/summary/YOUR_SESSION_ID
```

```json
{
  "session_id": "YOUR_SESSION_ID",
  "user_id": "user-123",
  "summary_text": "...",
  "primary_intent": "info_seeking",
  "risk_level": "low",
  "total_turns": 6
}
```

---

## Customize

Edit `step4_summarizer/summarizer.py` to:
- Change when summarization is triggered (currently: every 5 messages)
- Add new recommended actions for specific intents
- Adjust risk thresholds

---

## Next step

➡️ Summaries are accessible via the **Step 5** API endpoint `/summary/{session_id}`
