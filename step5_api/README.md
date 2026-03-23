# Step 5 — API Deployment (FastAPI + Redis)

**Location:** `step5_api/`  
**Where to run:** 🖥️ VM `192.168.0.184` — **ALREADY RUNNING** ✅  
**Goal:** Serve the full RAVEN pipeline as a REST API

---

## Services running on the VM

| Container | Port | Status |
|-----------|------|--------|
| `raven_api` — FastAPI | `8000` | ✅ Running |
| `raven_redis` — Redis | `6379` | ✅ Running |

---

## Check status

```bash
ssh litellm-tybo@192.168.0.184
# password: litellm-tybo123

sudo docker ps
```

Expected output:
```
CONTAINER ID   IMAGE           PORTS                    NAMES
xxxxxxxxxxxx   raven_api       0.0.0.0:8000->8000/tcp   raven_api
xxxxxxxxxxxx   redis:7-alpine  0.0.0.0:6379->6379/tcp   raven_redis
```

---

## API Endpoints

### `GET /health`
Check if the API is alive.

```bash
curl http://192.168.0.184:8000/health
```
```json
{"status": "ok", "version": "1.0.0"}
```

---

### `POST /chat`
Send a message and get an AI response.

```bash
curl -X POST http://192.168.0.184:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain quantum computing?",
    "session_id": "my-session-001",
    "language": "en"
  }'
```

**Response:**
```json
{
  "session_id": "my-session-001",
  "response": "Quantum computing uses quantum bits (qubits) which can exist in multiple states simultaneously...",
  "intent": "info_seeking",
  "tags": ["science", "technology"],
  "lang": "en",
  "confidence": 0.954,
  "pii_detected": false,
  "pii_types": [],
  "was_blocked": false,
  "risk_level": "low",
  "processing_time_ms": 1240
}
```

---

### `GET /summary/{session_id}`
Get the summary of a conversation session.

```bash
curl http://192.168.0.184:8000/summary/my-session-001
```

---

### `GET /stats`
View global API statistics.

```bash
curl http://192.168.0.184:8000/stats
```

---

## Start / Stop / Restart

```bash
ssh litellm-tybo@192.168.0.184
cd ~/LLM_TYBOO

# Start
sudo docker-compose up -d

# Stop
sudo docker-compose down

# Restart
sudo docker-compose restart

# Rebuild from scratch (after code changes)
sudo docker-compose down
sudo docker-compose up -d --build
```

---

## Update after code changes

```bash
# 1. On your local machine — push changes to GitHub
git add .
git commit -m "your changes"
git push

# 2. On the VM — pull and redeploy
ssh litellm-tybo@192.168.0.184
cd ~/LLM_TYBOO
git pull
sudo docker-compose down && sudo docker-compose up -d --build
```

---

## View logs

```bash
ssh litellm-tybo@192.168.0.184

# Live logs
sudo docker logs -f raven_api

# Last 50 lines
sudo docker logs raven_api --tail 50

# Redis logs
sudo docker logs raven_redis
```

---

## Environment variables

Configured in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis container hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `ravenpass` | Redis password |
| `INTENT_MODEL` | `step2_intent_classifier/models/raven_cnn.pkl` | Path to intent classifier |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM model to use |

---

## Notes

> **First startup:** The API downloads `Qwen2.5-0.5B-Instruct` (~1GB) from HuggingFace on first run. This may take 5-10 minutes. After that, it is cached in Docker volume `hf_cache`.

> **No GPU:** The VM runs the LLM on CPU only. Responses may take 5-30 seconds depending on message length.

> **After fine-tuning:** Replace `LLM_MODEL` in `docker-compose.yml` with your fine-tuned model path and restart.
