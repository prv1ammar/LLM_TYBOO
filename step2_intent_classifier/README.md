# Step 2 — Intent Classifier (Fine-tuning)

**Location:** `step2_intent_classifier/`  
**Where to run:** ☁️ Google Colab (GPU required)  
**Goal:** Fine-tune a multilingual model to classify user intents

---

## What it does

- Takes conversations from Step 1 as training data
- Fine-tunes `xlm-roberta-base` (or similar multilingual model)
- Classifies these intents from any user message:

| Intent | Description |
|--------|-------------|
| `info_seeking` | User wants information or explanation |
| `creative` | Creative writing, poetry, stories |
| `problem_solving` | Debugging, planning, fixing |
| `casual_chat` | Greetings, small talk |
| `sensitive` | Touchy or controversial topics |
| `harmful` | Dangerous or harmful requests |

- Saves the fine-tuned model to `models/qwen_intent_finetuned/`

---

## How to run on Google Colab

### Step-by-step

**1. Open Google Colab**
- Go to [colab.research.google.com](https://colab.research.google.com)
- Click **New Notebook**
- Go to **Runtime → Change runtime type → GPU (T4)**

**2. Upload your data**
```python
from google.colab import files
uploaded = files.upload()
# Upload: data/conversations_en.json, conversations_fr.json, etc.
```

**3. Upload the classifier script**
```python
from google.colab import files
uploaded = files.upload()
# Upload: step2_intent_classifier/classifier.py
```

**4. Install dependencies**
```bash
!pip install transformers datasets scikit-learn torch accelerate
```

**5. Run the classifier training**
```bash
!python classifier.py
```

**6. Download the trained model**
```python
import shutil
from google.colab import files
shutil.make_archive('qwen_intent_finetuned', 'zip', 'models/qwen_intent_finetuned')
files.download('qwen_intent_finetuned.zip')
```

---

## Transfer the model to the VM

After downloading from Colab, unzip and transfer to the VM:

```bash
# Unzip locally
unzip qwen_intent_finetuned.zip -d models/qwen_intent_finetuned

# Transfer to VM
scp -r models/qwen_intent_finetuned litellm-tybo@192.168.0.184:~/LLM_TYBOO/models/
# password: litellm-tybo123
```

---

## Test the classifier locally

```bash
cd c:\Users\info\Downloads\raven
venv\Scripts\activate
python step2_intent_classifier/classifier.py
```

---

## Files in this folder

| File | Description |
|------|-------------|
| `classifier.py` | Main classifier with training + inference |
| `classifier_v2.py` | Improved version with better accuracy |
| `neural_classifier.py` | Neural network-based approach |
| `improved_classifier.py` | Latest version with all improvements |

> **Tip:** Start with `improved_classifier.py` for best results.

---

## Next step

➡️ Upload trained model to VM → used automatically by **Step 5** (API)
