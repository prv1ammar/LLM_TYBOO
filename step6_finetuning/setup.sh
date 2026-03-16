#!/bin/bash
# =============================================================
# setup.sh — تثبيت كل شي وتشغيل pipeline
# شغّل: bash setup.sh
# =============================================================

echo "================================================"
echo "  تجهيز بيئة Fine-tuning Qwen"
echo "================================================"

# --- 1. تثبيت المكتبات ---
echo ""
echo "[1/3] 📦 تثبيت المكتبات..."
pip install -q \
    transformers==4.47.0 \
    torch \
    accelerate \
    bitsandbytes \
    peft \
    trl \
    datasets \
    sentencepiece \
    protobuf

echo "✅ تم تثبيت المكتبات"

# --- 2. تحقق من RAM ---
echo ""
echo "[2/3] 🔍 فحص الموارد..."
python3 -c "
import psutil, shutil
ram = psutil.virtual_memory()
disk = shutil.disk_usage('.')
print(f'   RAM المتاحة: {ram.available / 1024**3:.1f} GB / {ram.total / 1024**3:.1f} GB')
print(f'   Disk المتاح: {disk.free / 1024**3:.1f} GB')

# تحقق torch
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {\"✅ متاح\" if torch.cuda.is_available() else \"❌ غير متاح (CPU فقط)\"}')
" 2>/dev/null || echo "   ⚠️  pip install psutil"

# --- 3. شغّل pipeline ---
echo ""
echo "[3/3] 🚀 بدء توليد البيانات..."
python3 generate_with_qwen.py
