# 🚀 DevOps Deployment Guide - AGATE AI Platform

This is the **complete deployment checklist** for the DevOps team to bring the AGATE Enterprise AI Platform into production.

---

## 📋 Pre-Deployment Checklist

### 1️⃣ Server Preparation
- [ ] **Verify Server Specs**:
  - CPU: 24 cores available (platform will use max 10)
  - RAM: 124 GB available (platform will use max 24 GB)
  - Storage: 1 TB available (platform will use max 157 GB)
  - OS: Ubuntu 22.04 LTS or compatible Linux
- [ ] **Install Docker & Docker Compose**:
  ```bash
  # Install Docker
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  
  # Install Docker Compose
  sudo apt-get update
  sudo apt-get install docker-compose-plugin
  
  # Verify installation
  docker --version
  docker compose version
  ```

### 2️⃣ Clone Repository
```bash
cd /opt
sudo git clone https://github.com/prv1ammar/LLM_TYBOO.git agate
cd agate
```

### 3️⃣ Configure Environment Variables
- [ ] **Copy environment template**:
  ```bash
  cp .env.example .env
  ```
- [ ] **Edit `.env` file** and set the following (get from Project Owner via secure channel):
  ```bash
  nano .env
  ```
  
  **Required Variables**:
  ```bash
  JWT_SECRET_KEY=<64-char-random-string-from-owner>
  NGINX_API_KEY=<32-char-random-string-from-owner>
  ```
  
  **Optional (for cloud fallback)**:
  ```bash
  GROQ_API_KEY=<groq-api-key-if-provided>
  OPENAI_API_KEY=<openai-api-key-if-provided>
  ```

### 4️⃣ Create Data Directories
```bash
mkdir -p data/{ollama,qdrant,postgres,redis,prometheus,grafana,loki}
mkdir -p documents
chmod -R 755 data documents
```

---

## 🚢 Deployment Steps

### Step 1: Build and Start Services
```bash
# Build custom API container and start all services
docker compose up -d --build

# Verify all containers are running
docker compose ps
```

**Expected Output**: All services should show status "Up"

### Step 2: Initialize AI Model
```bash
# Pull the Qwen2.5-7B model (first time only, ~5GB download)
docker exec -it $(docker ps -qf "name=ollama") ollama pull qwen2.5:7b
```

**Wait Time**: 5-10 minutes depending on internet speed

### Step 3: Verify Health
```bash
# Check if Nginx gateway is responding
curl http://localhost/health

# Expected output: "healthy"
```

### Step 4: Access Admin Dashboard
- **URL**: `http://<server-ip>/admin/`
- **Default Credentials**: 
  - Username: `admin`
  - Password: `password123`
- **Action**: Login and verify the dashboard loads

---

## 🔍 Post-Deployment Verification

### ✅ Service Health Checks

Run these commands to verify each component:

```bash
# 1. Check all containers
docker compose ps

# 2. Check Ollama (LLM)
curl http://localhost:11434/api/tags

# 3. Check LiteLLM Gateway
curl http://localhost:4000/health

# 4. Check Qdrant Vector DB
curl http://localhost:6333/

# 5. Check Grafana
curl http://localhost:3000/api/health

# 6. View logs for any service
docker compose logs -f [service_name]
```

### ✅ Resource Usage Verification

```bash
# Monitor real-time resource usage
docker stats

# Verify CPU limits are enforced (should never exceed ~10 cores total)
# Verify RAM limits are enforced (should never exceed ~24GB total)
```

---

## 📊 Monitoring & Observability

### Access Points

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Admin Dashboard** | `http://<server-ip>/admin/` | Platform management |
| **API Docs** | `http://<server-ip>/api/docs` | Interactive API testing |
| **Grafana** | `http://<server-ip>/dashboard/` | Metrics & performance |
| **Prometheus** | `http://<server-ip>:9090` | Raw metrics data |
| **Qdrant UI** | `http://<server-ip>/qdrant/dashboard` | Vector database explorer |

### Setting Up Alerts

1. **Login to Grafana**: `http://<server-ip>/dashboard/`
2. **Navigate to**: Alerting → Alert Rules
3. **Create alerts for**:
   - CPU usage > 90%
   - RAM usage > 90%
   - Disk usage > 80%
   - API response time > 5s

---

## 🔒 Security Hardening (Production)

### 1️⃣ Change Default Credentials
```bash
# Edit src/api.py and update FAKE_USERS_DB
# Then rebuild:
docker compose up -d --build api
```

### 2️⃣ Configure Firewall
```bash
# Allow only necessary ports
sudo ufw allow 80/tcp    # Nginx gateway
sudo ufw allow 443/tcp   # HTTPS (when SSL is configured)
sudo ufw enable
```

### 3️⃣ Enable SSL/HTTPS (Recommended)
- Obtain SSL certificate (Let's Encrypt or company CA)
- Update `nginx.conf` with SSL configuration
- Restart Nginx: `docker compose restart nginx`

---

## 🔄 Maintenance Operations

### Update the Platform
```bash
cd /opt/agate
git pull origin main
docker compose up -d --build
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f ollama
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart api
```

### Backup Data
```bash
# Backup all persistent data
tar -czf agate-backup-$(date +%Y%m%d).tar.gz data/
```

### Scale Services (if needed)
```bash
# Increase API replicas from 2 to 4
docker compose up -d --scale api=4 --scale litellm=4
```

---

## 🆘 Troubleshooting

### Issue: Container won't start
```bash
# Check logs
docker compose logs [service_name]

# Check resource limits
docker stats
```

### Issue: Out of memory
```bash
# Reduce replicas
docker compose up -d --scale api=1 --scale litellm=1
```

### Issue: Slow responses
```bash
# Check if model is loaded
docker exec -it $(docker ps -qf "name=ollama") ollama list

# Monitor CPU usage
htop
```

### Issue: Disk full
```bash
# Check disk usage
df -h

# Clean Docker cache
docker system prune -a
```

---

## 📞 Support Contacts

- **Project Owner**: [Owner's contact info]
- **Platform Documentation**: See `README.md` and `MANUAL_GUIDE.md`
- **Architecture Details**: See `PRODUCTION_ROADMAP.md`

---

## ✅ Deployment Completion Checklist

- [ ] All containers running (`docker compose ps`)
- [ ] Qwen2.5-7B model downloaded
- [ ] Admin dashboard accessible
- [ ] Grafana monitoring configured
- [ ] Firewall rules applied
- [ ] SSL certificate installed (if applicable)
- [ ] Default credentials changed
- [ ] Backup strategy implemented
- [ ] Project Owner notified of successful deployment

---

**🎉 Once all items are checked, the AGATE platform is live and ready for production use!**
