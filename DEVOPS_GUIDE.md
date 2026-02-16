# DevOps Deployment Guide - Self-Hosted AI Infrastructure

**Project**: Reusable LLM Backend Service  
**Status**: Development Complete - Ready for Production  
**Handoff Date**: 2026-02-04

---

## 🏗️ Architecture Overview

The system consists of **two layers**:

1. **Infrastructure Layer (Docker)**
   - **vLLM**: LLM Inference Engine (DeepSeek-R1)
   - **TEI**: Embedding Engine (BGE-M3)
   - **LiteLLM**: Internal Gateway

2. **Application Layer (System Service)**
   - **Backend API**: FastAPI service (`src/backend_api.py`)
   - Handles job queues, RAG, and multi-tenancy
   - Exposes REST API on port `8888`

---

## 🚀 Deployment Steps

### Step 1: Deploy Infrastructure (Docker)

Follow standard Docker deployment:

```bash
cd /opt/llm-stack
docker-compose up -d
```

### Step 2: Deploy Backend Service (Systemd)

The backend API should run as a persistent system service, NOT inside Docker (to keep it closer to the metal and easier to debug/update).

#### 2.1 Create Systemd Service File
Create `/etc/systemd/system/llm-backend.service`:

```ini
[Unit]
Description=LLM Backend API Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llm-stack
EnvironmentFile=/opt/llm-stack/.env
ExecStart=/usr/bin/python3 /opt/llm-stack/src/backend_api.py
Restart=always
RestartSec=5

# Logging
StandardOutput=append:/var/log/llm-backend.log
StandardError=append:/var/log/llm-backend.error.log

[Install]
WantedBy=multi-user.target
```

#### 2.2 enable & Start Service

```bash
# Install dependencies first
pip install -r /opt/llm-stack/requirements.txt

# Start service
systemctl daemon-reload
systemctl enable llm-backend
systemctl start llm-backend

# Check status
systemctl status llm-backend
```

### Step 3: Nginx Configuration (Update)

Update Nginx to expose the new Backend API port (8888).

```nginx
server {
    listen 8888 ssl;
    server_name api.your-internal-domain.com;

    # ... SSL settings ...

    location / {
        proxy_pass http://localhost:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Increase timeout for long-running jobs if polling
        proxy_read_timeout 300;
    }
}
```

---

## 🔍 Monitoring

Monitor the backend logs specifically:

```bash
tail -f /var/log/llm-backend.log
```

---

## 🔐 Security Checklist

1. **API Key**: Ensure `API_KEY` in `.env` is set to a strong random string.
2. **Firewall**: Only expose port `443` (Nginx). Block 8888/4000/8000 externally.
3. **Internal Access**: Client apps should connect via the Nginx URL (e.g., `https://api.internal/`).

---

## 🔄 Updating the API

When developer pushes new code to `src/backend_api.py`:

```bash
cd /opt/llm-stack
git pull origin main
systemctl restart llm-backend
```
