# Step 2 — Intent Classifier (TextCNN)

**Location:** `step2_intent_classifier/`  
**Where to run:** ☁️ Google Colab (T4 GPU recommended)  
**Goal:** Train a multilingual TextCNN model that classifies user intent and assigns segmentation tags from any message in English, French, Arabic, or Darija.

---

## What it does

- Loads a balanced CSV dataset (100K rows) from Google Drive
- Enriches tags using the official segmentation taxonomy (`Segmentation_TybotSmartContact.xlsx`)
- Trains a custom **TextCNN** architecture with dual output heads (intent + tags)
- Classifies these 12 intents from any user message:

| Intent | Description |
|--------|-------------|
| `request` | User is asking for something specific |
| `interest` | User is exploring or curious about products |
| `negotiation` | User is trying to get a better deal or price |
| `confirmation` | User wants to verify or confirm something |
| `emotional_expression` | User expresses stress, frustration, or anxiety |
| `reminder` | User is following up on an unresolved issue |
| `acknowledgment` | User confirms they understood |
| `persuasion` | User is applying pressure or making a deal |
| `salutation` | Greetings and farewells |
| `social` | Small talk and casual conversation |
| `feedback` | User shares an opinion about the service |
| `follow_up` | User checks status of a previous request |

- Assigns segmentation tags from 350 official tags (demographics, behavioral, transactional, predictive)
- Saves the trained model as both `.pt` and `.pkl` 

---

## Architecture

```
Embedding(vocab=10,000, dim=64)
→ Conv1D(kernels=3,4,5 × 192 filters) → GlobalMaxPool → concat(576)
→ Dropout(0.35) → Linear(576→576) → ReLU → Dropout
├── intent_head : Linear(576 → 12)   → CrossEntropyLoss(label_smooth=0.1)
└── tags_head   : Linear(576 → N_TAGS) → BCEWithLogitsLoss
```

**Tokenizer:** char trigrams + word tokens, Arabic normalization, vocab=10K, seq_len=96  
**Languages:** English, French, Arabic (MSA), Moroccan Darija

---

## How to run

### Step-by-step

**1. Open Google Colab**
- Go to [colab.research.google.com](https://colab.research.google.com)
- Click **New Notebook**
- Go to **Runtime → Change runtime type → GPU (T4)**

**2. Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**3. Upload required files to your Drive**

Place these files in `/content/drive/MyDrive/RAVEN/`:
- `IIbalanced_shuffled_dataset.csv` — training data (100K rows)
- `Segmentation_TybotSmartContact.xlsx` — official tag taxonomy

**4. Upload the training script**
```python
from google.colab import files
uploaded = files.upload()
# Upload: raven_cnn.py
```

**5. Update paths at the top of `raven_cnn.py`**


**6. Install dependencies**
```bash
!pip install torch pandas scikit-learn numpy openpyxl
```

**7. Run the training**
```bash
!python raven_cnn.py
```

Training takes ~15–25 minutes on a T4 GPU.

**8. Output files saved to Drive**

| File | Description |
|------|-------------|
| `raven_cnn_torch.pt` | PyTorch state dict (for long-term use) |
| `raven_cnn_torch.pkl` | Full pickled model (for easy deployment) |

---



---

## Load and use the model

### From `.pkl` (easiest)
```python
import pickle, torch

with open("raven_cnn_torch.pkl", "rb") as f:
    payload = pickle.load(f)

model          = payload["model"].eval()
tok            = payload["tok"]
intent_encoder = payload["intent_encoder"]
TAG_VOCAB      = payload["tag_vocab"]
```

### From HuggingFace
```python
from huggingface_hub import hf_hub_download
import pickle

path = hf_hub_download("your_username/raven-intent-classifier", "raven_cnn_torch.pkl")
with open(path, "rb") as f:
    payload = pickle.load(f)

model          = payload["model"].eval()
tok            = payload["tok"]
intent_encoder = payload["intent_encoder"]
TAG_VOCAB      = payload["tag_vocab"]
```

### Run inference
```python
predict("I need to block my card right now")
predict("bghit n7awel flous l compte dyal khoya")
predict("أريد إيقاف بطاقتي فوراً")
predict("je voudrais réinitialiser mon mot de passe")
```

---

## Transfer model to VM

```bash
# Download .pkl from Drive or HuggingFace, then transfer to VM
scp raven_cnn_torch.pkl litellm-tybo@192.168.0.184:~/LLM_TYBOO/models/
# password: litellm-tybo123
```

---

## Test locally

```bash
cd c:\Users\admin\Desktop\NOUVEAU\LLM_TYBOO
venv\Scripts\activate
python step2_intent_classifier/raven_cnn.py
```

---

## Config reference

| Parameter | Value |
|-----------|-------|
| `VOCAB_SIZE` | 10,000 |
| `SEQ_LEN` | 96 |
| `EMBED_DIM` | 64 |
| `CNN_FILTERS` | 192 per kernel (576 total) |
| `EPOCHS` | 60 (early stopping, patience=10) |
| `BATCH_SIZE` | 256 |
| `LEARNING_RATE` | 2e-3 |
| `DROPOUT` | 0.35 |
| `AUG_FACTOR` | 4× (rare intents: 8×) |

---

## Files in this folder

| File | Description |
|------|-------------|
| `raven_cnn.py` | Full training + inference pipeline |
| `Segmentation_TybotSmartContact.xlsx` | Official tag taxonomy (350 tags) |
| `IIbalanced_shuffled_dataset.csv` | Balanced training dataset (100K rows) |

---

## Next step

➡️ Upload trained model to VM → used automatically by **Step 5** (API)
