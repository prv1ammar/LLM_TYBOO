# 👨‍💻 Developer Guide: Using Your AI Backend

**Role**: Developer / Consumer  
**Goal**: Run the stack locally and build apps using the SDK.

---

## 🚀 1. Quick Start (Local)

1. **Verify Config**: Ensure your `.env` file has your tokens.
2. **Start the Stack**:
   ```bash
   ./start.sh
   # This starts Docker (vLLM, TEI) AND the Python Backend API (port 8888)
   ```
3. **Check Health**:
   Open http://localhost:8888/health in your browser.

---

## 📦 2. How to Use the SDK

The core of this system is the **Python SDK**. You don't need to write raw API requests.

### Setup in Your OTHER Projects

1. **Copy the SDK File**:
   Copy `sdk/llm_client.py` from this repo into your other project's folder.

2. **Install Dependencies**:
   ```bash
   pip install requests python-dotenv
   ```

3. **Initialize the Client**:
   ```python
   from llm_client import LLMBackendClient

   # Point to your local backend
   client = LLMBackendClient(
       api_url="http://localhost:8888",
       api_key="your-secure-api-key-here" # Matches API_KEY in .env
   )
   ```

---

## ⚡ 3. Common Tasks

### 💬 Chat & Completion
```python
# Simple Chat
response = client.chat("Summarize this text: ...")

# Structured Generation (JSON)
# You can instruct the model to return JSON in the prompt
response = client.chat("Extract names as JSON: ...")
```

### 🧠 Embeddings
```python
# Single
vector = client.embed_single("Analysis text")

# Batch (Optimized)
vectors = client.embed(["Text 1", "Text 2", "Text 3"])
```

### 📚 RAG (Knowledge Base)
Use **Collections** to keep projects separate.

```python
# Project A: HR
client.rag_ingest(hr_docs, collection="hr_project")
client.rag_query("Holiday policy?", collection="hr_project")

# Project B: Finance
client.rag_ingest(finance_docs, collection="finance_project")
```

### ⏳ Async Jobs (Long Running)
For heavy tasks, don't wait. Submit a job.

```python
# 1. Submit
job_id = client.submit_job("batch_embed", {"texts": huge_list})

# 2. Check Later
status = client.get_job_status(job_id)
if status['status'] == 'completed':
    print(status['result'])
```

---

## 🛠️ 4. Debugging

- **Logs**: Check `backend.log` in the project root.
- **Docker Logs**: `docker-compose logs -f`
- **Restart**: `kill <pid>` (find in start.sh output) then run `./start.sh` again.
