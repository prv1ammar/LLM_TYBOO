#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting LLM_TYBOO Deployment..."

# 1. Create models directory
mkdir -p ~/models

# 2. Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ HF_TOKEN is not set. Downloads might fail if models are restricted."
fi

# 3. Download Models
echo "📥 Downloading models to ~/models..."

# 14B Q4_K_M
if [ ! -f ~/models/qwen2.5-14b-instruct-q4_k_m.gguf ]; then
    echo "⏳ Downloading Qwen2.5-14B Q4_K_M (~8.9GB)..."
    curl -L -f --retry 5 \
         -o ~/models/qwen2.5-14b-instruct-q4_k_m.gguf \
         "https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"
else
    echo "✅ 14B Model already exists."
fi

# 3B Q4_K_M
if [ ! -f ~/models/qwen2.5-3b-instruct-q4_k_m.gguf ]; then
    echo "⏳ Downloading Qwen2.5-3B Q4_K_M (~2GB)..."
    curl -L -f --retry 5 \
         -o ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \
         "https://huggingface.co/lmstudio-community/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.GGUF"
else
    echo "✅ 3B Model already exists."
fi

# 4. Start Docker Stack
echo "🐳 Starting Docker Compose stack..."
docker compose up -d --build

echo "✨ Deployment Finished!"
echo "📍 Dashboard: http://192.168.0.184:8501"
echo "📍 LiteLLM: http://192.168.0.184:4000"
echo "📍 n8n: http://192.168.0.184:5678"
