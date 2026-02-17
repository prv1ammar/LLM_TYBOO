# 🚀 Enterprise AI Platform: Manual Setup & Usage Guide

This guide provides step-by-step instructions to manually deploy, manage, and use your self-hosted AI infrastructure.

---

## 🛠️ 1. Initial Setup

### 1.1 Environment Variables
Before launching, you must configure your secrets.
1.  Copy `.env.example` to `.env`.
2.  Update the following keys:
    *   `HUGGING_FACE_HUB_TOKEN`: Your HF token (required for downloading models).
    *   `JWT_SECRET_KEY`: A long, random string for securing your login tokens.
    *   `NGINX_API_KEY`: The key used for the outer gateway protection (X-API-Key header).

### 1.2 Resource Preparation
Ensure your system has at least:
*   **GPU**: 24GB+ VRAM (Recommended for DeepSeek-70B-AWQ).
*   **Storage**: 100GB+ free space for models and vector data.
*   **RAM**: 32GB+ System RAM.

---

## 🚢 2. Deployment

Run the following command to build the custom API and start all services in the background:

```bash
docker-compose up -d --build --remove-orphans
```

### 🔍 Verify Status
Check if all containers are healthy:
```bash
docker-compose ps
```

---

## 📚 3. Data Ingestion (RAG)

To add your company documents to the knowledge base:

1.  **Add Files**: Place your PDFs, `.txt`, or `.md` files in the `documents/` directory.
2.  **Run Ingestor**:
    ```bash
    python src/ingest.py
    ```
    *This will chunk the text and store it in the Qdrant vector database.*

---

## 🔐 4. Accessing the system

### 4.1 Interactive API Docs (Swagger)
Access the auto-generated documentation to test your agents:
*   **URL**: `http://localhost:8888/docs` (or via Nginx on port 80/api/docs)

### 4.2 Authentication Flow
1.  **Login**: Use the `/token` endpoint with:
    *   **Username**: `admin`
    *   **Password**: `password123`
2.  **Use Token**: Copy the `access_token` and use it in the `Authorization: Bearer <token>` header for all other requests.

---

## 📊 5. Monitoring & Dashboards

| Service | Local URL | Description |
| :--- | :--- | :--- |
| **Nginx Gateway** | `http://localhost` | Main entry point (Port 80) |
| **Admin Dashboard** | `http://localhost/admin/` | Platform Control Center |
| **API Docs** | `http://localhost/api/docs` | Test Agents & RAG |
| **Grafana** | `http://localhost/dashboard/` | Visualize GPU & Token Usage |
| **Qdrant UI** | `http://localhost/qdrant/dashboard` | Inspect your vector data |
| **Prometheus** | `http://localhost:9090` | Raw metrics data |

---

## 🛠️ 6. Troubleshooting

*   **Logs**: Check Loki logs via Grafana or use docker commands:
    ```bash
    docker-compose logs -f [service_name]
    ```
*   **GPU Issues**: Run `nvidia-smi` to ensure Docker has access to your GPU.
*   **OOM Errors**: If services crash with "Out of Memory", reduce `replicas` in `docker-compose.yaml` or lower `gpu-memory-utilization` in the vLLM command.

---

## ✅ 7. Conclusion
Your platform is now ready. For any further architectural changes, refer to `PRODUCTION_ROADMAP.md`.
