# Quick Reference: What You Need to Deploy

## 🔑 The ONLY Thing You Must Do

### Get Your Hugging Face Token

**Why?** The system needs this to download AI models (DeepSeek-R1, BGE-M3) from Hugging Face.

**How to get it:**

1. Go to: https://huggingface.co/settings/tokens
2. Click the blue **"+ New token"** button
3. Enter:
   - **Name**: `llm-project-deployment`
   - **Type**: Select **"Read"**
4. Click **"Generate token"**
5. **Copy the token** - it looks like: `hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890`

![Hugging Face Token Guide](../../../.gemini/antigravity/brain/b7f0b55d-b34e-4cb0-881e-8a331e1c161a/huggingface_token_guide_1770204543135.png)

---

## 📝 Where to Add the Token

### File: `.env` (in your project root)

**Current state:**
```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
```

**What you need to do:**
Replace `your_token_here` with your actual token:

```bash
HUGGING_FACE_HUB_TOKEN=hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890
```

**Full example of `.env` file:**
```bash
# Hugging Face Token for downloading models
HUGGING_FACE_HUB_TOKEN=hf_XyZ123AbC456DeF789GhI012JkL345MnO678

# LiteLLM Configuration (you can leave these as-is for now)
LITELLM_URL=http://localhost:4000
LITELLM_KEY=sk-1234-change-this-in-production
```

---

## ✅ That's It!

**Files you need to edit:** Just **1 file** (`.env`)

**What you need to change:** Just **1 line** (the Hugging Face token)

---

## 🚀 After Adding the Token

You're ready to deploy! Run:

**Windows:**
```powershell
.\start.bat
```

**Linux/WSL:**
```bash
./start.sh
```

---

## 📚 For More Details

See `DEPLOYMENT_GUIDE.md` for:
- Hardware requirements
- Troubleshooting
- Production deployment
- Security hardening
