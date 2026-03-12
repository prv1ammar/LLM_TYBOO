"""
RAVEN — CPU-Optimized Fine-tuning Script
=========================================
Modified version of finetune.py for VM environments WITHOUT GPU.
Note: Training on CPU is significantly slower than GPU.

Usage:
  python scripts/finetune_cpu.py --stage both --dry-run
"""

import os, sys, json, time, argparse, shutil
from pathlib import Path
import torch

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  MODEL RESOLUTION
# ══════════════════════════════════════════════════════════════

PRIMARY_MODEL  = "Qwen/Qwen3-0.6B"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def resolve_model() -> tuple[str, str]:
    from huggingface_hub import model_info
    for model_id in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            info = model_info(model_id, timeout=10)
            print(f"[Model] ✅ Using: {model_id}")
            tag = "qwen3-0.6b" if "Qwen3" in model_id else "qwen2.5-0.5b"
            return model_id, f"raven-{tag}-cpu"
        except Exception as e:
            print(f"[Model] ⚠️  {model_id} unavailable ({e}) — trying fallback...")
    raise RuntimeError("Both primary and fallback models are unavailable.")

# ══════════════════════════════════════════════════════════════
#  LORA CONFIGS (CPU works with LoRA too)
# ══════════════════════════════════════════════════════════════

LORA_SFT = dict(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","v_proj"]) # Lighter for CPU

LORA_DPO = dict(r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","v_proj"])

# ══════════════════════════════════════════════════════════════
#  DATA LOADING (Shared with finetune.py)
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
        print("[Warning] DPO data not found. Run scripts/generate_training_data.py first.")
        return Dataset.from_list([])
    with open(dpo_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                pairs.append(json.loads(line))
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
    system = example.get("system", "You are RAVEN, a banking assistant.")
    prompt_msgs = [{"role":"system","content":system},{"role":"user","content":example["prompt"]}]
    chosen_msgs = prompt_msgs + [{"role":"assistant","content":example["chosen"]}]
    rejected_msgs = prompt_msgs + [{"role":"assistant","content":example["rejected"]}]
    return {
        "prompt":   tokenizer.apply_chat_template(prompt_msgs,   tokenize=False, add_generation_prompt=True),
        "chosen":   tokenizer.apply_chat_template(chosen_msgs,   tokenize=False, add_generation_prompt=False),
        "rejected": tokenizer.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False),
    }

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING (CPU Optimized)
# ══════════════════════════════════════════════════════════════

def load_base_model_cpu(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\n[Load] {model_id} on CPU (Full Precision/FP32)...")
    
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    # No BitsAndBytesConfig used here because it's GPU only
    mod = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map={"": "cpu"}, # Force CPU
        torch_dtype=torch.float32, 
        trust_remote_code=True
    )
    
    total = sum(p.numel() for p in mod.parameters())
    print(f"[Load] {total/1e6:.1f}M params  |  CPU FP32")
    return mod, tok

def apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model, TaskType
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, **cfg)
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()
    return model

# ══════════════════════════════════════════════════════════════
#  TRAINING STAGES
# ══════════════════════════════════════════════════════════════

def train_sft(model_id: str, out_name: str, epochs=1, batch=1, lr=1e-4, dry_run=False, samples=500):
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig

    n_samples = 100 if dry_run else samples
    sft_out = OUT_DIR / out_name / "sft"
    sft_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[SFT] CPU Stage 1 | Epochs: {epochs} Batch: {batch}")
    dataset = load_sft_dataset(n_samples)
    if len(dataset) == 0: return None
    split   = dataset.train_test_split(test_size=0.1, seed=42)

    mod, tok = load_base_model_cpu(model_id)
    mod = apply_lora(mod, LORA_SFT)

    fmt = lambda ex: format_sft_text(ex, tok)
    ds_train = split["train"].map(fmt, remove_columns=split["train"].column_names)
    ds_val   = split["test"].map(fmt,  remove_columns=split["test"].column_names)

    args = SFTConfig(
        output_dir = str(sft_out / "checkpoints"),
        num_train_epochs = epochs,
        per_device_train_batch_size = batch,
        gradient_accumulation_steps = 4,
        learning_rate = lr,
        use_cpu = True,
        bf16 = False,
        fp16 = False,
        logging_steps = 10,
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 2,
        report_to = "none",
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=mod, args=args,
        train_dataset=ds_train, eval_dataset=ds_val, processing_class=tok,
    )

    t0 = time.time()
    trainer.train()
    print(f"[SFT] ✅ Done in {(time.time()-t0)/60:.1f} min")

    adapter_path = str(sft_out / "lora_adapters")
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    
    merged_path = str(sft_out / "merged")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_path)
    tok.save_pretrained(merged_path)
    return merged_path

def train_dpo(sft_model_path: str, out_name: str, epochs=1, batch=1, lr=5e-5, dry_run=False, samples=200):
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import DPOTrainer, DPOConfig
    
    n_samples = 50 if dry_run else (samples // 2)
    dpo_out = OUT_DIR / out_name / "dpo"
    dpo_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[DPO] CPU Stage 2 | Epochs: {epochs}")
    dataset = load_dpo_dataset(n_samples)
    if len(dataset) == 0: return None
    split   = dataset.train_test_split(test_size=0.1, seed=42)

    tok = AutoTokenizer.from_pretrained(sft_model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(sft_model_path, device_map={"": "cpu"}, torch_dtype=torch.float32)
    mod = apply_lora(mod, LORA_DPO)

    fmt = lambda ex: format_dpo_example(ex, tok)
    ds_train = split["train"].map(fmt, remove_columns=split["train"].column_names)
    ds_val   = split["test"].map(fmt,  remove_columns=split["test"].column_names)

    dpo_config = DPOConfig(
        output_dir = str(dpo_out / "checkpoints"),
        num_train_epochs = epochs,
        per_device_train_batch_size = batch,
        gradient_accumulation_steps = 4,
        learning_rate = lr,
        use_cpu = True,
        bf16 = False,
        fp16 = False,
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 2,
        report_to = "none"
    )

    trainer = DPOTrainer(model=mod, ref_model=None, args=dpo_config, train_dataset=ds_train, eval_dataset=ds_val, processing_class=tok)
    
    t0 = time.time()
    trainer.train()
    print(f"[DPO] ✅ Done in {(time.time()-t0)/60:.1f} min")

    final_path = str(OUT_DIR / out_name / "final")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    return final_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",   default="both", choices=["sft","dpo","both"])
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--samples", type=int, default=500)
    args = p.parse_args()

    model_id, out_name = resolve_model()
    sft_merged = None

    if args.stage in ("sft", "both"):
        sft_merged = train_sft(model_id, out_name, dry_run=args.dry_run, samples=args.samples)
    
    if args.stage in ("dpo", "both"):
        path = sft_merged or str(OUT_DIR / out_name / "sft" / "merged")
        if os.path.exists(path):
            train_dpo(path, out_name, dry_run=args.dry_run, samples=args.samples)
        else:
            print(f"[Error] SFT model not found at {path}")

if __name__ == "__main__":
    main()
