# 🦅 RAVEN
## Repetition-Aware, Verified, Extracted & Noted

> Multilingual LLM Pipeline — Anti-répétition · Guardrails · Intent Classification · Summarization

---

## Vue d'ensemble / نظرة عامة

RAVEN est un pipeline LLM intelligent qui résout le problème des **réponses répétitives** dans les conversations.  
Il supporte **4 langues** : العربية · الدارجة · Français · English

```
User Message
    │
    ▼
┌─────────────────────────────────────┐
│  Step 3: Guardrails                 │  ← PII scan (RIB, CIN, Nom...)
│          Redis Save                 │  ← Profil user sécurisé
│          JSONL Audit                │  ← Log compliance
└────────────────┬────────────────────┘
                 │ masked message
                 ▼
┌─────────────────────────────────────┐
│  Step 2: Intent Classifier          │  ← TextCNN (PyTorch) rapide et léger
│          → intent + tags            │  ← question_info / complaint / ...
└────────────────┬────────────────────┘
                 │ intent + tags
                 ▼
┌─────────────────────────────────────┐
│  Step 1: LLM (Qwen 0.5B / 1.5B)    │  ← Fine-tuné sur data anti-répétition (Step 6)
│          Anti-repetition engine     │  ← repetition_penalty + score check
└────────────────┬────────────────────┘
                 │ response
                 ▼
┌─────────────────────────────────────┐
│  Step 3 (bis): Scan response        │  ← Vérif PII dans la réponse LLM
└────────────────┬────────────────────┘
                 │
    ┌────────────┘  (background)
    ▼
┌─────────────────────────────────────┐
│  Step 4: Summarizer                 │  ← Résumé + intentions + tags + risk
│          → Redis + JSONL            │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 5: FastAPI                    │  ← /chat /session /summary /health
└─────────────────────────────────────┘
```

---

## Structure du Projet

```
raven/
├── step1_data_generation/
│   └── generate.py          # Dataset 200k samples (4 langues)
│
├── step2_intent_classifier/
│   ├── raven_cnn.py         # Architecture et entraînement réseau CNN
│   ├── neural_classifier.py # Wrapper API pour l'inférence
│   └── models/              # Modèle exporté raven_cnn.pkl
│
├── step3_guardrails/
│   └── guardrails.py        # PII scanner + Redis + JSONL audit
│
├── step4_summarizer/
│   └── summarizer.py        # Conversation summarizer + risk level
│
├── step5_api/
│   └── api.py               # FastAPI orchestration (tous les steps)
│
├── step6_finetuning/
│   ├── finetune_qwen_colab.ipynb # Notebook d'entraînement LoRA
│   └── general_chatbot_data.jsonl # Dataset formaté conversation
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 3. Generate dataset (Step 1)
cd step1_data_generation
python generate.py
# → data/train.jsonl (~160k), val.jsonl, test.jsonl

# 4. Train intent classifier (Step 2)
cd ../step2_intent_classifier
python raven_cnn.py
# → models/raven_cnn.pkl (Modèle + Vocabulaire)

# 5. Fine-tune your LLM (Step 6 - Optionnel)
cd ../step6_finetuning
# Lancer le notebook ou script Python pour obtenir l'adapter LoRA.

# 6. Launch API (Step 5 — orchestre Steps 1,2,3,4)
cd ../step5_api
python api.py
# → http://localhost:8000
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/chat` | Message principal — pipeline complet |
| GET | `/session/{user_id}/{session_id}` | Historique + profil user |
| GET | `/summary/{session_id}` | Résumé de conversation |
| POST | `/feedback` | Feedback sur une réponse |
| GET | `/health` | Status de tous les composants |

### Exemple `/chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "bghit n3ref 3la compte dyali",
    "user_id": "user_123",
    "session_id": "sess_abc"
  }'
```

**Response:**
```json
{
  "session_id": "sess_abc",
  "response": "ymken t9da ma3loumat 3la compte dyalek...",
  "intent": "question_info",
  "tags": ["account_info", "faq"],
  "lang": "darija",
  "confidence": 0.94,
  "pii_detected": false,
  "pii_types": [],
  "was_blocked": false,
  "risk_level": "low",
  "processing_time_ms": 234.5
}
```

---

## Step 2 — Intentions Détectées

| Intent | Description | Tags automatiques |
|--------|-------------|-------------------|
| `question_info` | Demande d'information | `faq`, `account_info` |
| `complaint` | Réclamation / insatisfaction | `frustrated`, `requires_human` |
| `transaction` | Action / modification compte | `action_request`, `form_needed` |
| `support` | Support technique | `support`, `technical_issue` |
| `off_topic` | Hors sujet | — |
| `emergency` | Urgence (compte piraté...) | `urgent`, `security_risk`, `requires_human` |

---

## Step 3 — PII Détectés & Masqués

| Type | Exemple | Masque |
|------|---------|--------|
| RIB | MA76 2001 1900... | `[RIB_MASQUÉ]` |
| CIN | AB123456 | `[CIN_MASQUÉ]` |
| Téléphone | 0661234567 | `[TEL_MASQUÉ]` |
| Email | user@bank.ma | `[EMAIL_MASQUÉ]` |
| Nom complet | M. Youssef Benali | `[NOM_MASQUÉ]` |
| Carte bancaire | 4532 1234 5678 | `[CARTE_MASQUÉE]` |

**Redis Schema (par user):**
```
raven:user:{user_id}     → UserProfile JSON (TTL 24h)
raven:history:{session}  → List of turns (masked only, max 50)
raven:summary:{session}  → ConversationSummary JSON (TTL 7j)
```

---

## Step 4 — Résumé de Conversation

```json
{
  "session_id": "sess_abc",
  "user_id": "user_123",
  "lang": "en",
  "created_at": 1718000000.0,
  "summary_text": "L'utilisateur a demandé des informations sur son compte...",
  "summary_lang": "french",
  "primary_intent": "question_info",
  "all_intents": ["question_info", "complaint"],
  "all_tags": ["banking", "frustrated", "requires_human"],
  "total_turns": 4,
  "user_turns": 2,
  "assistant_turns": 2,
  "avg_user_msg_len": 45.5,
  "risk_level": "high",
  "pii_types_detected": [],
  "was_blocked": false,
  "recommended_actions": ["Transmettre au service réclamations"],
  "needs_human_takeover": true,
  "escalation_reason": "Client frustré avec intent complaint détecté"
}
```

---

## Step 6 — Fine-Tuning (LoRA)

Ce step se charge d'affiner le modèle principal (Qwen) pour s'assurer qu'il ne produit pas de répétitions et prend la "personnalité" souhaitée par l'entreprise.

- **Fichiers clés :** `finetune_qwen_colab.ipynb`, `general_chatbot_data.jsonl`.
- **Technique :** LoRA (Low-Rank Adaptation) permettant un entraînement rapide et économe.
- L'adapter produit dans cette étape est ensuite chargé dynamiquement par **Step 1/5** au démarrage de l'API.

---

## Modèles Utilisés

| Role | Modèle | Taille | Usage |
|------|--------|--------|-------|
| Chat principal | Qwen2.5-0.5B-Instruct + LoRA | ~1GB | Réponses rapides |
| Chat qualité | Qwen2.5-1.5B-Instruct + LoRA | ~3GB | Réponses complexes |
| Intent classifier | TextCNN (PyTorch Custom) | ~6MB | Step 2 (Très rapide) |
| Summarizer | Qwen2.5-1.5B (rule fallback) | ~3GB | Step 4 |

---

## Sécurité & Compliance

- ✅ **Jamais de PII brut stocké** — hash SHA256 uniquement dans Redis
- ✅ **JSONL audit** de chaque interaction (masqué)
- ✅ **Prompt injection** détectée et bloquée
- ✅ **TTL automatique** — données supprimées après 24h (sessions) / 7j (summaries)
- ✅ **Scan double** — user message ET réponse LLM

---

*RAVEN v1.0 — Built with Qwen2.5 + FastAPI + Redis*
