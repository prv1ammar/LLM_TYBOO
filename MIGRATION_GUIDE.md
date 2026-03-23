# 🦅 RAVEN: Company Laptop Deployment Guide

Follow these steps to migrate the RAVEN project from your local machine to your company workstation.

## 🛠️ Phase 1: Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/prv1ammar/LLM_TYBOO.git raven
   cd raven
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   source venv/Scripts/activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Phase 2: Hugging Face Authentication
To download the base model fast (3GB), use your token:
```bash
export HF_TOKEN="VOTRE_TOKEN_ICI"
```

## 🚀 Phase 3: Hardware Verification (CPU vs GPU)
The current `step5_api/api.py` is optimized for **CPU**. 
If your company laptop has a **NVIDIA GPU (CUDA)**, you can make it fly even faster by changing these lines in `step5_api/api.py`:
*   Change `torch_dtype=torch.float32` → `torch_dtype=torch.float16`
*   Change `device_map={"": "cpu"}` → `device_map="auto"`

## 🏁 Phase 4: Local Live Test
Run this command to verify both the **CNN Intent** and **LLM/LoRA** models:
```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from step2_intent_classifier.neural_classifier import CNNClassifier
from step5_api.api import LLMEngine, Config

print('\n--- RAVEN LIVE TEST ---')
# 1. Test CNN
print('[1/2] Loading Intent CNN...')
cnn = CNNClassifier.load('step2_intent_classifier/models/raven_cnn.pkl')
print(f'Done! Intent Match: {cnn.predict(\"بغيت نبدل رقم الهاتف\")}')

# 2. Test LLM + LoRA
print('\n[2/2] Loading Qwen + LoRA Adapter...')
llm = LLMEngine(Config.LLM_MODEL, Config.ADAPTER_PATH)
resp = llm.generate([], 'Hello!')
print(f'AI Response: {resp}')
"
```

## 📡 Phase 5: Launch API Server
Once the test passes, run the main server:
```bash
python step5_api/api.py
```

## 🧪 Phase 6: API Testing (Postman)
*   **Endpoint:** `http://localhost:8000/chat`
*   **Method:** `POST`
*   **JSON Body:**
    ```json
    {
      "message": "بغيت نبدل رقم الهاتف",
      "user_id": "laptop_move_test"
    }
    ```
