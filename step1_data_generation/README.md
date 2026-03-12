# Step 1 — Data Generation

**Location:** `step1_data_generation/`  
**Where to run:** 💻 Local Machine  
**Goal:** Generate synthetic multilingual training conversations

---

## What it does

- Creates realistic conversations across all intents:
  - `info_seeking` — User asking for information
  - `creative` — Creative writing, brainstorming
  - `problem_solving` — Debugging, fixing, planning
  - `casual_chat` — Small talk, greetings
  - `sensitive` — Touchy topics (handled carefully)
  - `harmful` — Dangerous requests (blocked)
- Supports 4 languages: **Arabic, Darija, French, English**
- Inserts fake PII (names, phones, emails, credit cards) for guardrail testing
- Saves data as JSON files in `data/`

---

## How to run

```bash
# 1. Activate your virtual environment
cd c:\Users\info\Downloads\raven
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / Mac

# 2. Run the generator
python step1_data_generation/generate.py
```

---

## Expected output

```
data/
  conversations_ar.json     ← Arabic
  conversations_dart.json   ← Darija (Moroccan Arabic)
  conversations_fr.json     ← French
  conversations_en.json     ← English
```

---

## Verify the output

```bash
python -c "

import json
for lang in ['en', 'fr', 'ar']:
    data = json.load(open(f'data/conversations_{lang}.json'))
    print(f'{lang}: {len(data)} samples')
    print('Sample:', data[0]['messages'][0]['content'][:80])
    print()
    
"
```

---

## Customize

Edit `step1_data_generation/generate.py` to:
- **Change number of samples** → find `N_SAMPLES = 100` and increase it
- **Add new topics** → add to the `TOPICS` dictionary per language
- **Add new languages** → add a new language block following the same pattern

---

## Next step

➡️ Take the generated `data/` folder to **Step 2** (Intent Classifier training on Google Colab)
