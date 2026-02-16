#!/bin/bash

# Quick start script for the self-hosted AI stack
# This script checks prerequisites and starts the infrastructure

set -e

echo "🚀 Self-Hosted AI Stack - Quick Start"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check .env
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your HUGGING_FACE_HUB_TOKEN"
    exit 1
fi

echo "🐳 Starting Infrastructure (vLLM, TEI, LiteLLM)..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services..."
sleep 10

echo ""
echo "🔥 Starting Backend API Service..."
echo "   Running on http://localhost:8888"
echo "   Logs will be written to backend.log"

# Install requirements if needed
pip install -r requirements.txt > /dev/null 2>&1

# Start the backend API in background
nohup python3 src/backend_api.py > backend.log 2>&1 &
API_PID=$!

echo "✅ Backend API running under PID $API_PID"
echo ""
echo "📚 Usage:"
echo "   - API Docs:     http://localhost:8888/docs"
echo "   - Health Check: http://localhost:8888/health"
echo "   - Client SDK:   sdk/llm_client.py"
echo ""
echo "🛑 To stop: kill $API_PID && docker-compose down"
