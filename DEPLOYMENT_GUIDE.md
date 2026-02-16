# Complete Deployment Guide - Step by Step

This guide will walk you through **exactly** what you need to deploy this self-hosted AI stack.

---

## 🔑 Step 1: Get Your Hugging Face Token

### What is it?
A Hugging Face token is like a password that allows the system to download AI models (DeepSeek-R1, BGE-M3) from Hugging Face's servers.

### How to get it:

1. **Go to Hugging Face**: https://huggingface.co/
2. **Create an account** (if you don't have one) - it's free
3. **Go to Settings → Access Tokens**: https://huggingface.co/settings/tokens
4. **Click "New token"** or "Create new token"
5. **Fill in**:
   - Name: `llm-project-deployment`
   - Type: Select **"Read"** (you only need read access to download models)
6. **Click "Generate token"**
7. **Copy the token** - it will look like: `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890`

⚠️ **IMPORTANT**: Save this token somewhere safe! You won't be able to see it again.

---

## 📝 Step 2: Add the Token to Your Project

### File to edit: `.env`

**Location**: `c:\Users\info\Desktop\llm\.env`

**What to change**:

**BEFORE** (current state):
```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
```

**AFTER** (replace with your actual token):
```bash
HUGGING_FACE_HUB_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890
```

**Example with a real token**:
```bash
# Hugging Face Token for downloading models
HUGGING_FACE_HUB_TOKEN=hf_XyZ123AbC456DeF789GhI012JkL345MnO678

# LiteLLM Configuration
LITELLM_URL=http://localhost:4000
LITELLM_KEY=sk-1234-change-this-in-production
```

---

## 🔒 Step 3: Secure Your LiteLLM API Key (Optional but Recommended)

### File to edit: `.env`

**What to change**:

Replace `sk-1234-change-this-in-production` with a strong random key.

**Generate a secure key** (run this in PowerShell):
```powershell
-join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
```

**Example**:
```bash
LITELLM_KEY=sk-aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3
```

---

## 🖥️ Step 4: Hardware Requirements Check

### Minimum Requirements:

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 22.04+) or Windows with WSL2 |
| **GPU** | NVIDIA GPU with 48GB+ VRAM (e.g., 2x RTX 3090, 1x A100) |
| **RAM** | 32GB+ |
| **Storage** | 200GB+ free space (for models) |
| **Docker** | Docker 24.0+ with nvidia-container-toolkit |

### For Smaller GPUs (24GB or less):

If you have a single RTX 3090/4090 (24GB), you need to enable **quantization**.

**File to edit**: `docker-compose.yaml`

**Find this section** (around line 25):
```yaml
command: >
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B
  --tensor-parallel-size 1
  --max-model-len 4096
```

**Change to**:
```yaml
command: >
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B
  --tensor-parallel-size 1
  --max-model-len 4096
  --quantization awq
```

This reduces memory usage by ~50% with minimal quality loss.

---

## 🚀 Step 5: Pre-Deployment Checklist

Before you run the system, verify:

- [ ] ✅ You have a Hugging Face token
- [ ] ✅ Token is added to `.env` file (replace `your_token_here`)
- [ ] ✅ You have Docker installed with GPU support
- [ ] ✅ You have WSL2 enabled (if on Windows)
- [ ] ✅ You have at least 200GB free disk space
- [ ] ✅ Your GPU has at least 48GB VRAM (or quantization is enabled)

---

## 🎯 Step 6: Deploy the System

### On Windows (with WSL2):

1. **Open PowerShell** as Administrator
2. **Navigate to project**:
   ```powershell
   cd C:\Users\info\Desktop\llm
   ```
3. **Run the start script**:
   ```powershell
   .\start.bat
   ```

### On Linux or WSL2 directly:

1. **Open terminal**
2. **Navigate to project**:
   ```bash
   cd /mnt/c/Users/info/Desktop/llm
   ```
3. **Make script executable**:
   ```bash
   chmod +x start.sh
   ```
4. **Run the start script**:
   ```bash
   ./start.sh
   ```

### Manual deployment (if scripts don't work):

```bash
# 1. Start Docker services
docker-compose up -d

# 2. Wait for services to start (this takes 5-10 minutes on first run)
docker-compose logs -f

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Check health
python src/health_check.py
```

---

## ⏱️ What to Expect During First Deployment

### Timeline:

1. **Docker pulls images**: ~5 minutes (downloading vLLM, TEI, LiteLLM containers)
2. **Model downloads**: ~30-60 minutes (downloading DeepSeek-R1 ~70GB + BGE-M3 ~2GB)
3. **Model loading**: ~5 minutes (loading models into GPU memory)
4. **Total first run**: ~45-75 minutes

### Subsequent runs:
- Models are cached, so startup takes only ~5 minutes

---

## ✅ Step 7: Verify Everything Works

### Run health check:
```bash
python src/health_check.py
```

**Expected output**:
```
✅ vLLM is healthy
✅ TEI is healthy
✅ LiteLLM is healthy
✅ LiteLLM models available: {...}
✅ All services are healthy!
```

### Test the system:

1. **Test basic agent**:
   ```bash
   python src/main.py
   ```

2. **Test embeddings**:
   ```bash
   python src/embeddings.py
   ```

3. **Test RAG pipeline**:
   ```bash
   python src/rag_pipeline.py
   ```

---

## 🔧 Troubleshooting

### Error: "Could not authenticate with Hugging Face"
- ✅ Check that your token is correct in `.env`
- ✅ Make sure there are no extra spaces or quotes around the token
- ✅ Verify the token has "Read" permissions

### Error: "CUDA out of memory"
- ✅ Enable quantization in `docker-compose.yaml` (see Step 4)
- ✅ Reduce `--max-model-len` to 2048 or 1024
- ✅ Use a smaller model (e.g., Llama-3.1-8B)

### Error: "nvidia-smi not found"
- ✅ Install NVIDIA drivers in WSL2
- ✅ Install nvidia-container-toolkit in Docker

### Services won't start
```bash
# Check Docker logs
docker-compose logs vllm
docker-compose logs tei
docker-compose logs litellm

# Restart services
docker-compose down
docker-compose up -d
```

---

## 📊 Production Deployment Checklist

For deploying to a production server:

- [ ] Change `LITELLM_KEY` to a strong random key
- [ ] Setup SSL/TLS with nginx or Traefik
- [ ] Configure firewall rules (only allow internal network access)
- [ ] Setup monitoring (Prometheus + Grafana)
- [ ] Configure automatic backups
- [ ] Setup log rotation
- [ ] Document your deployment for your team

---

## 🎓 Summary: Files You Need to Edit

| File | What to Change | Example |
|------|----------------|---------|
| **`.env`** | `HUGGING_FACE_HUB_TOKEN` | `hf_XyZ123...` |
| **`.env`** | `LITELLM_KEY` (optional) | `sk-random32chars...` |
| **`docker-compose.yaml`** | Add `--quantization awq` (if GPU < 48GB) | See Step 4 |

**That's it!** Only 1-2 files to edit, and you're ready to deploy.

---

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review Docker logs: `docker-compose logs -f`
3. Verify GPU access: `nvidia-smi`
4. Check disk space: `df -h`

---

**You're now ready to deploy! 🚀**
