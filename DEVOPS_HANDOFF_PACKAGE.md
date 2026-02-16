# 👷 DevOps Deployment Package

**Project**: AI Backend Service (Microservice Architecture)  
**Components**: Docker Infrastructure + Systemd Python Service  
**Ports**: 8000 (vLLM), 8080 (TEI), 4000 (Gateway), **8888 (Main API)**

---

## 📋 1. Infrastructure Setup

### Server Requirements
- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA A100/H100 or RTX 3090/4090
- **Software**: Docker, NVIDIA Container Toolkit, Python 3.10+

### Step 1: Deploy Docker Containers
The core AI models run in Docker to manage CUDA dependencies.

```bash
cd /opt/llm-stack
docker-compose up -d
```
*Wait ~45 mins for first-time model downloads.*

---

## ⚙️ 2. Service Deployment (Systemd)

The application logic (Job queues, RAG, API) runs as a native system service for performance and persistence.

### Step 1: Install Dependencies
```bash
pip3 install -r /opt/llm-stack/requirements.txt
```

### Step 2: Create Service File
Create `/etc/systemd/system/llm-backend.service`:

```ini
[Unit]
Description=LLM Backend API
After=docker.service network.target
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llm-stack
EnvironmentFile=/opt/llm-stack/.env
ExecStart=/usr/bin/python3 /opt/llm-stack/src/backend_api.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/llm-backend.log
StandardError=append:/var/log/llm-backend.error.log

[Install]
WantedBy=multi-user.target
```

### Step 3: Enable & Start
```bash
systemctl daemon-reload
systemctl enable llm-backend
systemctl start llm-backend
```

---

## 🔒 3. Security & Networking

### Firewall (UFW)
Only expose Nginx ports. Block everything else.
```bash
ufw allow 80
ufw allow 443
ufw deny 8888  # Block direct API access
ufw deny 8000  # Block vLLM
ufw deny 4000  # Block LiteLLM
```

### Nginx Reverse Proxy
Expose the API securely via Domain/SSL.

```nginx
server {
    listen 443 ssl;
    server_name ai.internal.your-company.com;

    # SSL Config...

    location / {
        proxy_pass http://localhost:8888;
        proxy_set_header Host $host;
        # 5 min timeout for long polling requests
        proxy_read_timeout 300; 
    }
}
```

---

## 🔄 4. Maintenance

### Logs
- **App Logs**: `tail -f /var/log/llm-backend.log`
- **Infrastructure Logs**: `docker-compose logs -f`

### Updates
To deploy new code:
```bash
cd /opt/llm-stack
git pull
systemctl restart llm-backend
```

### Backups
Backup the `/opt/llm-stack/.env` file and `config/` directory.
