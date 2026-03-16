"""
=============================================================
مولّد بيانات محلي — Qwen يولّد بياناته بنفسه
Local Data Generator using Qwen (no API needed)
=============================================================
المتطلبات:
  pip install transformers torch accelerate

الموارد المستخدمة:
  - RAM: ~8-12GB (Qwen2.5-1.5B بـ 8bit)
  - CPU: 10 cores
  - Storage: ~3GB للنموذج + البيانات
=============================================================
"""

import json
import os
import time
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =============================================
# إعداد
# =============================================
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"  # غيّر لـ 3B إذا عندك RAM كافية
OUTPUT_FILE  = "general_chatbot_data.jsonl"
TARGET       = 2000   # عدد المحادثات المطلوبة
BATCH_SAVE   = 10     # احفظ كل 10 محادثات

# =============================================
# تحميل النموذج مرة واحدة
# =============================================
print(f"⏳ جاري تحميل {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU → float32
    device_map="cpu",
    low_cpu_mem_usage=True,
    load_in_8bit=False,          # ← غيّر لـ True إذا عندك bitsandbytes مثبّت
)
model.eval()
print("✅ النموذج جاهز!\n")

# =============================================
# دالة التوليد الأساسية
# =============================================
def generate(messages: list[dict], max_new_tokens: int = 400) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # استخرج الرد فقط (بدون الـ prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =============================================
# 40 موضوع متنوع
# =============================================
TOPICS = [
    # يومي
    "casual greetings and small talk",
    "asking for advice about a personal decision",
    "planning a trip or vacation",
    "discussing food and cooking",
    "talking about movies or TV shows",
    "sports and fitness",
    "hobbies and free time activities",
    "weather and seasons",
    "family relationships",
    "personal goals and habits",
    # عمل
    "job hunting and CV advice",
    "workplace stress and conflicts",
    "starting a small business",
    "time management",
    "learning a new skill",
    "writing a professional email",
    "salary and promotions",
    "remote work tips",
    # تقنية
    "fixing a phone or laptop problem",
    "choosing the right software or app",
    "online safety and passwords",
    "understanding AI and technology",
    "social media tips",
    "internet connection issues",
    # صحة
    "general wellness questions",
    "dealing with stress and anxiety",
    "sleep problems",
    "exercise routines",
    "eating habits and nutrition",
    # تعليم
    "understanding a historical event",
    "science explained simply",
    "tips for learning a language",
    "study motivation and techniques",
    # خدمات
    "customer complaint and resolution",
    "understanding a bill or subscription",
    "returning a product",
    "asking for a recommendation",
    # محادثات صعبة
    "user sends vague unclear message",
    "user changes topic suddenly",
    "user asks many questions at once",
    "user disagrees with the answer",
]

STYLES = [
    "short, 2 turns only",
    "3 turns, user asks follow-up",
    "4 turns, casual and friendly",
    "3 turns, user is confused at first",
    "3 turns, user is in a hurry",
    "4 turns, user asks for more details",
]

# =============================================
# System prompt للنموذج أثناء التوليد
# =============================================
GENERATOR_SYSTEM = """You are a data generator. Generate a realistic chatbot conversation.
Output ONLY valid JSON. No explanation, no markdown, no extra text.

JSON format:
{
  "topic": "...",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Rules for the assistant in the conversation:
- Sound natural and human, never robotic
- Never say "As an AI..." or "I'm just a chatbot"
- Keep responses conversational, not bullet-pointed
- If user is vague, ask ONE clarifying question
- Remember what was said earlier
- Vary response length naturally"""

# =============================================
# دالة توليد محادثة واحدة
# =============================================
def generate_conversation(topic: str, style: str) -> dict | None:
    user_prompt = f'Generate a chatbot conversation about: "{topic}"\nStyle: {style}'
    
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]
    
    raw = generate(messages, max_new_tokens=500)
    
    # محاولة تحليل JSON
    try:
        # نظّف إذا جاء مع markdown
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
            if raw.startswith("json"):
                raw = raw[4:]
        # خذ أول { إلى آخر }
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(raw[start:end])
        if "conversation" in data and len(data["conversation"]) >= 2:
            return data
    except json.JSONDecodeError:
        pass
    
    return None

# =============================================
# تحويل لصيغة ChatML
# =============================================
TRAINING_SYSTEM = """You are a helpful, friendly, and knowledgeable assistant.
You engage naturally, remember context, adapt to any topic, ask clarifying questions when needed,
and admit uncertainty honestly without being robotic or overly formal."""

def to_chatml(conv_data: dict) -> str:
    lines = [f"<|im_start|>system\n{TRAINING_SYSTEM}<|im_end|>"]
    for turn in conv_data["conversation"]:
        lines.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
    return "\n".join(lines)

# =============================================
# Pipeline الرئيسي
# =============================================
def main():
    print("=" * 55)
    print("🤖 Qwen يولّد بياناته بنفسه")
    print(f"   الهدف: {TARGET} محادثة")
    print(f"   النموذج: {MODEL_NAME}")
    print("=" * 55)

    out_path = Path(OUTPUT_FILE)

    # تابع من حيث وقفت
    done = 0
    if out_path.exists():
        done = sum(1 for _ in open(out_path))
        print(f"📂 موجود: {done} محادثة — نكمّل\n")

    errors = 0
    buffer = []

    with open(out_path, "a", encoding="utf-8") as f:
        while done < TARGET:
            topic = random.choice(TOPICS)
            style = random.choice(STYLES)

            print(f"[{done+1}/{TARGET}] 📝 {topic[:45]}...", end=" ", flush=True)
            t0 = time.time()

            conv = generate_conversation(topic, style)

            if conv:
                entry = {
                    "text":  to_chatml(conv),
                    "topic": conv.get("topic", topic),
                }
                buffer.append(json.dumps(entry, ensure_ascii=False))
                done += 1
                errors = 0
                elapsed = time.time() - t0
                print(f"✅ ({elapsed:.1f}s)")

                # احفظ كل BATCH_SAVE محادثات
                if len(buffer) >= BATCH_SAVE:
                    f.write("\n".join(buffer) + "\n")
                    f.flush()
                    buffer = []
            else:
                errors += 1
                print(f"❌ فشل ({errors})")
                if errors >= 10:
                    print("⏸️ 10 أخطاء — انتظر 5 ثواني وكمّل")
                    time.sleep(5)
                    errors = 0

        # احفظ الباقي
        if buffer:
            f.write("\n".join(buffer) + "\n")

    print(f"\n✅ اكتملت! {done} محادثة في {OUTPUT_FILE}")
    split_dataset()


def split_dataset(
    input_file: str = None,
    train_ratio: float = 0.9
):
    input_file = input_file or OUTPUT_FILE
    lines = open(input_file, encoding="utf-8").readlines()
    random.shuffle(lines)
    n = int(len(lines) * train_ratio)
    
    with open("train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(lines[:n])
    with open("eval.jsonl", "w", encoding="utf-8") as f:
        f.writelines(lines[n:])
    
    print(f"📁 train.jsonl: {n} | eval.jsonl: {len(lines)-n}")


if __name__ == "__main__":
    main()
