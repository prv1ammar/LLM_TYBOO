"""
=============================================================
Fine-tuning Qwen2.5-1.5B بـ QLoRA
=============================================================
شغّل: python finetune_qwen.py
=============================================================
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================
# ⚙️ إعدادات — غيّر هنا فقط
# =============================================
DATA_FILE   = r"C:\Users\info\Downloads\raven\step6_finetuning\general_chatbot_data.jsonl"
OUTPUT_DIR  = r"C:\Users\info\Downloads\raven\step6_finetuning\qwen-finetuned"
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"

# hyperparameters
EPOCHS          = 3
BATCH_SIZE      = 1      # CPU → 1 دائماً
GRAD_ACCUM      = 8      # يحاكي batch size = 8
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 512    # زيد لـ 1024 إذا عندك RAM كافية
LORA_RANK       = 16
SAVE_STEPS      = 100
LOG_STEPS       = 10

# =============================================
# 1. تحميل وتقسيم البيانات
# =============================================
print("📂 تحميل البيانات...")

lines = Path(DATA_FILE).read_text(encoding="utf-8").strip().split("\n")
samples = [json.loads(l) for l in lines if l.strip()]

# تقسيم train/eval
split = int(len(samples) * 0.9)
train_samples = samples[:split]
eval_samples  = samples[split:]

train_dataset = Dataset.from_list(train_samples)
eval_dataset  = Dataset.from_list(eval_samples)

print(f"   ✅ Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# =============================================
# 2. تحميل النموذج والـ tokenizer
# =============================================
print(f"\n⏳ تحميل {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU → float32
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
print("   ✅ النموذج جاهز")

# =============================================
# 3. إعداد LoRA
# =============================================
print("\n🔧 إعداد LoRA...")

lora_config = LoraConfig(
    r               = LORA_RANK,
    lora_alpha      = LORA_RANK * 2,
    target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout    = 0.05,
    bias            = "none",
    task_type       = TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"   ✅ Parameters قابلة للتدريب: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# =============================================
# 4. إعداد التدريب
# =============================================
training_args = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    fp16                        = False,    # CPU → False
    bf16                        = False,    # CPU → False
    logging_steps               = LOG_STEPS,
    eval_strategy               = "steps",
    eval_steps                  = SAVE_STEPS,
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 2,        # احفظ آخر 2 checkpoints فقط
    load_best_model_at_end      = True,
    report_to                   = "none",
    dataset_text_field          = "text",
    max_seq_length              = MAX_SEQ_LEN,
    packing                     = False,
)

# =============================================
# 5. Trainer
# =============================================
trainer = SFTTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
)

# =============================================
# 6. بدء التدريب
# =============================================
print("\n🚀 بدء التدريب...")
print(f"   Epochs: {EPOCHS}")
print(f"   Total steps: {len(train_dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)}")
print(f"   Output: {OUTPUT_DIR}")
print("-" * 50)

trainer.train()

# =============================================
# 7. حفظ النموذج النهائي
# =============================================
print("\n💾 حفظ النموذج النهائي...")
final_path = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"   ✅ محفوظ في: {final_path}")

# =============================================
# 8. اختبار سريع بعد التدريب
# =============================================
print("\n🧪 اختبار سريع...")

from peft import PeftModel
from transformers import pipeline

test_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
)
test_model = PeftModel.from_pretrained(test_model, final_path)
test_model.eval()

def chat(user_message: str) -> str:
    messages = [
        {"role": "system",    "content": "You are a helpful, friendly assistant."},
        {"role": "user",      "content": user_message},
    ]
    text    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = test_model.generate(
            **inputs,
            max_new_tokens      = 200,
            do_sample           = True,
            temperature         = 0.7,
            top_p               = 0.9,
            repetition_penalty  = 1.1,
            pad_token_id        = tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()

test_questions = [
    "Hi! How are you?",
    "Can you help me write a professional email?",
    "What's the best way to learn a new skill?",
]

for q in test_questions:
    print(f"\n👤 {q}")
    print(f"🤖 {chat(q)}")

print("\n✅ التدريب اكتمل! النموذج جاهز في:", final_path)
