# Step 6 — LLM Fine-tuning (SFT + DPO)

**Location:** `step6_finetuning/`  
**Where to run:** ☁️ Google Colab (GPU required — T4 minimum, A100 recommended)  
**Goal:** Fine-tune `Qwen2.5-0.5B-Instruct` using SFT then DPO alignment

---

## What it does

### Phase 1 — SFT (Supervised Fine-Tuning)
- Trains the model to respond well across all domains (coding, science, creativity, etc.)
- Uses the data generated in Step 1
- Applies **LoRA** (Low-Rank Adaptation) for efficient fine-tuning on limited GPU memory

### Phase 2 — DPO (Direct Preference Optimization)
- Aligns the model preferences: teaches it to **prefer good answers over bad ones**
- Uses "chosen/rejected" pairs from `data/dpo_data.json`
- Makes the model safer and more helpful

---

## Files in this folder

| File | Description |
|------|-------------|
| `RAVEN_Finetuning_SFT_DPO.ipynb` | Main Google Colab notebook (recommended) |
| `scripts/generate_training_data.py` | Generate SFT + DPO data from scratch |
| `scripts/finetune.py` | Training script (SFT or DPO mode) |
| `scripts/eval.py` | Evaluate the fine-tuned model |
| `configs/sft.yaml` | SFT training configuration |
| `configs/dpo.yaml` | DPO training configuration |

---

## How to run — Option A: Notebook (recommended)

1. **Open Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Click **Upload** → upload `RAVEN_Finetuning_SFT_DPO.ipynb`

2. **Set GPU runtime**
   - `Runtime` → `Change runtime type` → **GPU**
   - Choose `T4` (free) or `A100` (Colab Pro)

3. **Generate training data locally first**
   ```bash
   # On your local machine
   cd c:\Users\info\Downloads\raven
   venv\Scripts\activate
   python step6_finetuning/scripts/generate_training_data.py
   # Creates: data/sft_data.json and data/dpo_data.json
   ```

4. **Upload data to Colab**
   ```python
   from google.colab import files
   uploaded = files.upload()
   # Upload: data/sft_data.json, data/dpo_data.json
   ```

5. **Run all cells** in the notebook

---

## How to run — Option B: Scripts directly

```python
# In a Colab cell

# Install dependencies
!pip install transformers datasets trl peft accelerate bitsandbytes

# Clone repository
!git clone https://github.com/prv1ammar/LLM_TYBOO.git
%cd LLM_TYBOO

# Upload your data files first, then:

# Phase 1 - SFT
!python step6_finetuning/scripts/finetune.py \
  --config step6_finetuning/configs/sft.yaml

# Phase 2 - DPO
!python step6_finetuning/scripts/finetune.py \
  --config step6_finetuning/configs/dpo.yaml
```

---

## Download the trained model from Colab

```python
import shutil
from google.colab import files

# Zip the model
shutil.make_archive('raven_finetuned', 'zip', 'models/raven_finetuned')

# Download to your PC
files.download('raven_finetuned.zip')
```

---

## Transfer model to the VM

```bash
# 1. Unzip on your local machine
unzip raven_finetuned.zip -d models/raven_finetuned

# 2. Copy to VM
scp -r models/raven_finetuned litellm-tybo@192.168.0.184:~/LLM_TYBOO/models/
# password: litellm-tybo123
```

---

## Activate the fine-tuned model in the API

Edit `docker-compose.yml` on the VM:

```bash
ssh litellm-tybo@192.168.0.184
cd ~/LLM_TYBOO
nano docker-compose.yml
```

Change this line:
```yaml
# Before
- LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct

# After
- LLM_MODEL=models/raven_finetuned
```

Restart the API:
```bash
sudo docker-compose down && sudo docker-compose up -d --build
```

---

## Training configuration

### SFT (`configs/sft.yaml`)
```yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
dataset: data/sft_data.json
output_dir: models/raven_sft
num_epochs: 3
batch_size: 4
learning_rate: 2e-4
lora_r: 16
lora_alpha: 32
```

### DPO (`configs/dpo.yaml`)
```yaml
model_name: models/raven_sft       # ← starts from SFT model
dataset: data/dpo_data.json
output_dir: models/raven_finetuned
num_epochs: 1
beta: 0.1
learning_rate: 5e-5
```

---

## Estimated training time

| GPU | SFT Duration | DPO Duration |
|-----|-------------|--------------|
| T4 (16GB) | ~45 min | ~20 min |
| A100 (40GB) | ~15 min | ~8 min |
| V100 (16GB) | ~35 min | ~15 min |

> **Tip:** Use Colab Pro for longer sessions and better GPUs. The free tier may disconnect during long training runs.

---

## Evaluate the model

```bash
!python step6_finetuning/scripts/eval.py \
  --model models/raven_finetuned \
  --test_data data/conversations_en.json
```

---

## Next step

➡️ Transfer the fine-tuned model to the VM and update **Step 5** (API) to use it
