# RAVEN — Step 6 : Fine-tuning 2 Stages (SFT + DPO)

Fine-tuning **Qwen3-0.6B** en 2 stages sur 10 GB de données bancaires multilingues.
Fallback automatique vers **Qwen2.5-0.5B** si Qwen3-0.6B est indisponible.

---

## Pipeline

```
Qwen3-0.6B (frozen 4-bit NF4)
        ↓
  ┌─ STAGE 1 — SFT ──────────────────────────────┐
  │  LoRA r=16 sur 10 GB conversations            │
  │  3 epochs · lr=2e-4 · batch=32               │
  │  ~5-7h T4 Colab                               │
  └──────────────────────────────────────────────┘
        ↓ merge
  raven-qwen3-0.6b/sft/merged
        ↓
  ┌─ STAGE 2 — DPO ──────────────────────────────┐
  │  LoRA r=8 sur 30k paires chosen/rejected      │
  │  1 epoch · lr=5e-5 · beta=0.1                │
  │  ~1-2h T4 Colab                               │
  └──────────────────────────────────────────────┘
        ↓ merge
  raven-qwen3-0.6b/final  ← deploy dans RAVEN API
```

---

## Démarrage rapide — Google Colab

1. Ouvre `RAVEN_Finetuning_SFT_DPO.ipynb` dans Colab
2. Runtime → **GPU T4**
3. Upload `RAVEN_project_v2.zip`
4. Exécute les cellules dans l'ordre

---

## CLI local

```bash
pip install -r requirements.txt

# Générer les données (10 GB SFT + 500 MB DPO)
python scripts/generate_training_data.py

# Dry run complet (test pipeline, ~5 min)
python scripts/finetune.py --stage both --dry-run

# Stage 1 — SFT
python scripts/finetune.py --stage sft --epochs-sft 3

# Stage 2 — DPO
python scripts/finetune.py --stage dpo --epochs-dpo 1

# Les deux enchaînés ✅
python scripts/finetune.py --stage both

# Évaluation
python scripts/eval.py --model outputs/raven-qwen3-0.6b/final --full

# Comparaison base vs SFT vs DPO
python scripts/eval.py --compare raven-qwen3-0.6b
```

---

## Structure des fichiers

```
step6_finetuning/
├── RAVEN_Finetuning_SFT_DPO.ipynb     ← Notebook Colab prêt
├── requirements.txt
├── configs/
│   ├── sft.yaml                        ← Hyperparamètres SFT
│   └── dpo.yaml                        ← Hyperparamètres DPO + beta
├── scripts/
│   ├── generate_training_data.py       ← 10 GB SFT + 30k DPO pairs
│   ├── finetune.py                     ← Pipeline SFT→DPO avec fallback auto
│   └── eval.py                         ← 16 cas test + comparaison stages
├── data/                               ← Généré automatiquement
│   ├── sft_ar.jsonl                    ← ~2.5 GB
│   ├── sft_darija.jsonl                ← ~2.5 GB
│   ├── sft_fr.jsonl                    ← ~2.5 GB
│   ├── sft_en.jsonl                    ← ~2.5 GB
│   └── dpo_all.jsonl                   ← ~500 MB (30k paires)
└── outputs/
    └── raven-qwen3-0.6b/              (ou raven-qwen2.5-0.5b si fallback)
        ├── sft/
        │   ├── lora_adapters/          ← ~100 MB
        │   └── merged/                 ← ~800 MB
        └── final/                      ← Modèle deploy-ready ✅
```

---

## Résultats attendus

| Stage | GPU T4 | Val Loss | Quality |
|-------|--------|----------|---------|
| Base (Qwen3-0.6B) | — | — | ~45% |
| After SFT | ~5-7h | ~0.9 | ~88% |
| After DPO | ~1-2h | — | ~95% |

## Pourquoi SFT + DPO ?

- **SFT seul** : apprend à imiter les bonnes réponses, mais peut quand même produire des réponses médiocres si le prompt est ambigu
- **DPO** : apprend explicitement à *préférer* les bonnes réponses ET *rejeter* les mauvaises (vagues, non-banking, sans urgence sur emergency...)
- Résultat : +7-15% de qualité sur les cas limites (off-topic refus, emergency avec 🚨, darija authentique)
