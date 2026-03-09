"""
RAVEN — 2-Stage Fine-tuning : SFT → DPO
=========================================
Stage 1 — SFT  : Supervised Fine-Tuning sur 10 GB de conversations
Stage 2 — DPO  : Direct Preference Optimization pour aligner le comportement

Modèle principal   : Qwen3-0.6B  (Qwen/Qwen3-0.6B)
Fallback automatique: Qwen2.5-0.5B si Qwen3-0.6B indisponible
GPU cible          : Google Colab T4 (15 GB) ou A100
Méthode            : QLoRA 4-bit NF4

Pipeline:
  Qwen3-0.6B (frozen 4-bit)
    → LoRA r=16 (SFT stage)
    → merge SFT adapters
    → LoRA r=8  (DPO stage)
    → merge final
    → raven-qwen3-0.6b (deploy)

Usage:
  python finetune.py --stage sft              # Stage 1 uniquement
  python finetune.py --stage dpo              # Stage 2 (après SFT)
  python finetune.py --stage both             # Les deux enchaînés ✅
  python finetune.py --stage sft  --dry-run   # Test 500 samples
  python finetune.py --stage both --epochs-sft 2 --epochs-dpo 1
"""

import os, sys, json, time, argparse, shutil
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  MODEL RESOLUTION — Qwen3-0.6B avec fallback Qwen2.5-0.5B
# ══════════════════════════════════════════════════════════════

PRIMARY_MODEL  = "Qwen/Qwen3-0.6B"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_NAME    = "raven-0.6b"

def resolve_model() -> tuple[str, str]:
    """
    Tente de charger Qwen3-0.6B.
    Si indisponible (HF Hub timeout, quota, etc.) → fallback vers Qwen2.5-0.5B.
    Retourne (model_id, actual_name).
    """
    from huggingface_hub import model_info
    for model_id in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            info = model_info(model_id, timeout=10)
            print(f"[Model] ✅ Using: {model_id}  (id={info.id})")
            tag = "qwen3-0.6b" if "Qwen3" in model_id else "qwen2.5-0.5b"
            return model_id, f"raven-{tag}"
        except Exception as e:
            print(f"[Model] ⚠️  {model_id} unavailable ({e}) — trying fallback...")
    raise RuntimeError("Both primary and fallback models are unavailable.")

# ══════════════════════════════════════════════════════════════
#  QLORA CONFIGS
# ══════════════════════════════════════════════════════════════

LORA_SFT = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

LORA_DPO = dict(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"])  # DPO: lighter LoRA

# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_sft_dataset(n_samples=None):
    from datasets import Dataset
    all_convs = []
    for fpath in sorted(DATA_DIR.glob("sft_*.jsonl")):
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    all_convs.append(json.loads(line))
                    count += 1
                    if n_samples and len(all_convs) >= n_samples: break
                except: pass
        print(f"  {fpath.name}: {count:,}")
        if n_samples and len(all_convs) >= n_samples: break
    print(f"[SFT Data] Total: {len(all_convs):,}")
    return Dataset.from_list(all_convs)


def load_dpo_dataset(n_samples=None):
    from datasets import Dataset
    pairs = []
    dpo_file = DATA_DIR / "dpo_all.jsonl"
    if not dpo_file.exists():
        raise FileNotFoundError(
            f"DPO data not found: {dpo_file}\n"
            "Run first: python scripts/generate_training_data.py --dpo-only"
        )
    with open(dpo_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                pairs.append(obj)
                if n_samples and len(pairs) >= n_samples: break
            except: pass
    print(f"[DPO Data] {len(pairs):,} preference pairs")
    return Dataset.from_list(pairs)


def format_sft_text(example, tokenizer):
    msgs = []
    for turn in example["conversations"]:
        role = {"human":"user","gpt":"assistant","system":"system"}.get(turn["from"], turn["from"])
        msgs.append({"role":role,"content":turn["value"]})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_dpo_example(example, tokenizer):
    """Format DPO pair into (prompt, chosen, rejected) with chat template."""
    system = example.get("system", "You are RAVEN, a banking assistant.")
    prompt_msgs = [
        {"role":"system","content":system},
        {"role":"user","content":example["prompt"]},
    ]
    chosen_msgs = prompt_msgs + [{"role":"assistant","content":example["chosen"]}]
    rejected_msgs = prompt_msgs + [{"role":"assistant","content":example["rejected"]}]
    prompt_text   = tokenizer.apply_chat_template(prompt_msgs,   tokenize=False, add_generation_prompt=True)
    chosen_text   = tokenizer.apply_chat_template(chosen_msgs,   tokenize=False, add_generation_prompt=False)
    rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False)
    return {
        "prompt":   prompt_text,
        "chosen":   chosen_text,
        "rejected": rejected_text,
    }

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_base_model(model_id: str, bf16=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    print(f"\n[Load] {model_id} in 4-bit NF4 QLoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mod = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    mod = prepare_model_for_kbit_training(mod)
    total = sum(p.numel() for p in mod.parameters())
    print(f"[Load] {total/1e6:.1f}M params  |  4-bit NF4")
    return mod, tok


def apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model, TaskType
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, **cfg)
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()
    return model

# ══════════════════════════════════════════════════════════════
#  STAGE 1 — SFT
# ══════════════════════════════════════════════════════════════

def train_sft(model_id: str, out_name: str,
              epochs=3, batch=4, lr=2e-4,
              dry_run=False, wandb_project="raven"):
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import wandb

    n_samples = 500 if dry_run else None
    sft_out = OUT_DIR / out_name / "sft"
    sft_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STAGE 1 — SFT  |  {model_id}")
    print(f"  Epochs: {epochs}  Batch: {batch}  LR: {lr}")
    print(f"  {'DRY RUN (500 samples)' if dry_run else 'FULL TRAINING'}")
    print(f"{'='*60}")

    # Data
    dataset = load_sft_dataset(n_samples)
    split   = dataset.train_test_split(test_size=0.02, seed=42)

    # Model
    mod, tok = load_base_model(model_id)
    mod = apply_lora(mod, LORA_SFT)

    # Format
    fmt = lambda ex: format_sft_text(ex, tok)
    ds_train = split["train"].map(fmt, remove_columns=split["train"].column_names)
    ds_val   = split["test"].map(fmt,  remove_columns=split["test"].column_names)

    if not dry_run:
        wandb.init(project=wandb_project, name=f"{out_name}-sft-ep{epochs}",
                   config={"stage":"sft","model":model_id,"epochs":epochs,"lr":lr})

    args = TrainingArguments(
        output_dir              = str(sft_out / "checkpoints"),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch,
        per_device_eval_batch_size  = batch,
        gradient_accumulation_steps = 8,
        learning_rate           = lr,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.03,
        weight_decay            = 0.001,
        max_grad_norm           = 0.3,
        bf16                    = True,
        logging_steps           = 25,
        save_strategy           = "steps",
        save_steps              = 500 if not dry_run else 50,
        eval_strategy           = "steps",
        eval_steps              = 500 if not dry_run else 50,
        save_total_limit        = 2,
        load_best_model_at_end  = True,
        report_to               = "wandb" if not dry_run else "none",
        group_by_length         = True,
        dataloader_num_workers  = 2,
    )
    trainer = SFTTrainer(
        model=mod, args=args,
        train_dataset=ds_train, eval_dataset=ds_val, tokenizer=tok,
        dataset_text_field="text", max_seq_length=2048, packing=True,
    )

    t0 = time.time()
    print(f"\n[SFT] Training started...")
    trainer.train()
    print(f"[SFT] ✅ Done in {(time.time()-t0)/60:.1f} min")

    # Save SFT adapters
    adapter_path = str(sft_out / "lora_adapters")
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"[SFT] Adapters saved → {adapter_path}")

    # Merge SFT → intermediate model
    merged_path = str(sft_out / "merged")
    print(f"[SFT] Merging LoRA → {merged_path}")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_path, safe_serialization=True)
    tok.save_pretrained(merged_path)
    print(f"[SFT] ✅ SFT merged model saved")

    if not dry_run: wandb.finish()
    return merged_path

# ══════════════════════════════════════════════════════════════
#  STAGE 2 — DPO
# ══════════════════════════════════════════════════════════════

def train_dpo(sft_model_path: str, out_name: str,
              epochs=1, batch=2, lr=5e-5,
              dry_run=False, wandb_project="raven"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    import wandb

    n_samples = 200 if dry_run else None
    dpo_out = OUT_DIR / out_name / "dpo"
    dpo_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STAGE 2 — DPO  |  base: {sft_model_path}")
    print(f"  Epochs: {epochs}  Batch: {batch}  LR: {lr}")
    print(f"  {'DRY RUN (200 pairs)' if dry_run else 'FULL DPO'}")
    print(f"{'='*60}")

    # Data
    dataset = load_dpo_dataset(n_samples)
    split   = dataset.train_test_split(test_size=0.05, seed=42)

    # Load SFT-merged model in 4-bit for DPO
    print(f"\n[DPO] Loading SFT model from {sft_model_path}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mod = AutoModelForCausalLM.from_pretrained(
        sft_model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    mod = prepare_model_for_kbit_training(mod)
    mod = apply_lora(mod, LORA_DPO)

    # Format DPO dataset
    fmt = lambda ex: format_dpo_example(ex, tok)
    ds_train = split["train"].map(fmt, remove_columns=split["train"].column_names)
    ds_val   = split["test"].map(fmt,  remove_columns=split["test"].column_names)

    if not dry_run:
        wandb.init(project=wandb_project, name=f"{out_name}-dpo-ep{epochs}",
                   config={"stage":"dpo","sft_base":sft_model_path,"epochs":epochs,"lr":lr})

    dpo_config = DPOConfig(
        output_dir              = str(dpo_out / "checkpoints"),
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch,
        per_device_eval_batch_size  = batch,
        gradient_accumulation_steps = 8,
        learning_rate           = lr,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.05,
        bf16                    = True,
        logging_steps           = 10,
        save_steps              = 200 if not dry_run else 50,
        eval_steps              = 200 if not dry_run else 50,
        save_total_limit        = 2,
        load_best_model_at_end  = True,
        report_to               = "wandb" if not dry_run else "none",
        beta                    = 0.1,   # DPO temperature — 0.1 = moderate alignment
        max_length              = 1024,
        max_prompt_length       = 512,
    )

    trainer = DPOTrainer(
        model=mod, ref_model=None,  # ref_model=None → uses implicit reference (implicit DPO)
        args=dpo_config,
        train_dataset=ds_train, eval_dataset=ds_val, tokenizer=tok,
    )

    t0 = time.time()
    print(f"\n[DPO] Training started...")
    trainer.train()
    print(f"[DPO] ✅ Done in {(time.time()-t0)/60:.1f} min")

    # Save DPO adapters
    adapter_path = str(dpo_out / "lora_adapters")
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)

    # Merge DPO → final model
    final_path = str(OUT_DIR / out_name / "final")
    print(f"[DPO] Merging → final model: {final_path}")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_path, safe_serialization=True)
    tok.save_pretrained(final_path)
    print(f"[DPO] ✅ FINAL MODEL saved → {final_path}")

    if not dry_run: wandb.finish()
    return final_path

# ══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(stage="both", epochs_sft=3, epochs_dpo=1,
                 batch_sft=4, batch_dpo=2,
                 lr_sft=2e-4, lr_dpo=5e-5,
                 dry_run=False, wandb_project="raven-finetuning"):

    model_id, out_name = resolve_model()

    print(f"\n{'='*60}")
    print(f"  🦅 RAVEN Fine-tuning Pipeline")
    print(f"  Model   : {model_id}")
    print(f"  Output  : {out_name}")
    print(f"  Stage   : {stage.upper()}")
    print(f"  DryRun  : {dry_run}")
    print(f"{'='*60}")

    sft_merged = str(OUT_DIR / out_name / "sft" / "merged")
    final_path = str(OUT_DIR / out_name / "final")

    if stage in ("sft", "both"):
        sft_merged = train_sft(model_id, out_name,
            epochs=epochs_sft, batch=batch_sft, lr=lr_sft,
            dry_run=dry_run, wandb_project=wandb_project)

    if stage in ("dpo", "both"):
        # If running DPO-only, check SFT model exists
        if stage == "dpo" and not Path(sft_merged).exists():
            raise FileNotFoundError(
                f"SFT merged model not found: {sft_merged}\n"
                "Run stage SFT first: python finetune.py --stage sft"
            )
        final_path = train_dpo(sft_merged, out_name,
            epochs=epochs_dpo, batch=batch_dpo, lr=lr_dpo,
            dry_run=dry_run, wandb_project=wandb_project)

    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete!")
    print(f"  SFT model  : {sft_merged}")
    if stage in ("dpo","both"):
        print(f"  FINAL model: {final_path}")
    print(f"\n  Next: python eval.py --model {final_path if stage!='sft' else sft_merged}")
    print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="RAVEN 2-Stage SFT+DPO Fine-tuning")
    p.add_argument("--stage",       default="both",  choices=["sft","dpo","both"])
    p.add_argument("--epochs-sft",  type=int,   default=3)
    p.add_argument("--epochs-dpo",  type=int,   default=1)
    p.add_argument("--batch-sft",   type=int,   default=4)
    p.add_argument("--batch-dpo",   type=int,   default=2)
    p.add_argument("--lr-sft",      type=float, default=2e-4)
    p.add_argument("--lr-dpo",      type=float, default=5e-5)
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--wandb",       default="raven-finetuning")
    args = p.parse_args()

    run_pipeline(
        stage       = args.stage,
        epochs_sft  = args.epochs_sft,
        epochs_dpo  = args.epochs_dpo,
        batch_sft   = args.batch_sft,
        batch_dpo   = args.batch_dpo,
        lr_sft      = args.lr_sft,
        lr_dpo      = args.lr_dpo,
        dry_run     = args.dry_run,
        wandb_project = args.wandb,
    )

if __name__ == "__main__":
    main()
