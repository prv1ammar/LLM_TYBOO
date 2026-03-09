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
cd c:\Users\info\Downloads\raven
venv\Scripts\activate
python step4_summarizer/summarizer.py
```

### Expected output

```json
{
  "session_id": "demo-session",
  "summary": "User asked about Python programming and scientific concepts.",
  "dominant_intent": "info_seeking",
  "risk_level": "low",
  "recommended_actions": [
    "Continue helping the user",
    "No escalation needed"
  ],
  "pii_detected": false,
  "message_count": 4,
  "languages_detected": ["en"]
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
  "summary": "...",
  "risk_level": "low",
  "message_count": 6
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
