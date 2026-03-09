# Step 3 — Guardrails (PII Scanner)

**Location:** `step3_guardrails/`  
**Where to run:** ✅ Runs automatically inside the API (VM)  
**Goal:** Detect and mask Personally Identifiable Information (PII)

---

## What it does

Every user message passes through the guardrails **before** reaching the LLM:

1. Scans the message for sensitive personal data
2. Masks detected PII with placeholder tokens
3. Returns the cleaned message to the API pipeline
4. Logs which PII types were found

---

## Supported PII types

| PII Type | Example | Replaced by |
|----------|---------|-------------|
| Phone number | `0612345678` | `[PHONE]` |
| Email address | `user@gmail.com` | `[EMAIL]` |
| Credit card | `4111 1111 1111 1111` | `[CREDIT_CARD]` |
| National ID | `AB123456` | `[ID]` |
| Person name (NER) | `John Smith` | `[NAME]` |

---

## How to test it manually

```bash
cd c:\Users\info\Downloads\raven
venv\Scripts\activate
python step3_guardrails/guardrails.py
```

### Expected output

```
=== PII Scanner Demo ===

Input:  "My email is user@gmail.com and my phone is 0612345678"
Output: "My email is [EMAIL] and my phone is [PHONE]"
PII found: ['email', 'phone']

Input:  "My card number is 4111 1111 1111 1111"
Output: "My card number is [CREDIT_CARD]"
PII found: ['credit_card']
```

---

## How it works in the API

```
User Message
    ↓
[Guardrails: PII Scanner]  ← Step 3
    ↓ (cleaned message)
[Intent Classifier]        ← Step 2
    ↓
[LLM Response]             ← Step 5
```

No changes needed — it runs completely automatically.

---

## Customize PII patterns

Edit `step3_guardrails/guardrails.py`:

```python
# Add a new PII pattern
PII_PATTERNS = {
    "passport": r"\b[A-Z]{2}[0-9]{7}\b",   # ← add your pattern here
    ...
}

MASK_TOKENS = {
    "passport": "[PASSPORT]",   # ← add your mask token here
    ...
}
```

---

## Next step

➡️ Guardrails feeds cleaned messages into **Step 4** (Summarizer) and **Step 5** (API/LLM)
