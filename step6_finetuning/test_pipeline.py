"""
test_pipeline.py — تحقق سريع قبل تشغيل pipeline كامل
شغّل: python3 test_pipeline.py
يولّد 3 محادثات فقط كاختبار (~5 دقائق)
"""
import json, random, time, sys

# =============================================
# 1. تحقق من المكتبات
# =============================================
print("🔍 فحص المكتبات...", flush=True)
missing = []
for lib in ["transformers", "torch", "accelerate"]:
    try:
        __import__(lib)
        print(f"   ✅ {lib}")
    except ImportError:
        print(f"   ❌ {lib} — مش مثبت")
        missing.append(lib)

if missing:
    print(f"\n❌ ثبّت أولاً: pip install {' '.join(missing)}")
    sys.exit(1)

# =============================================
# 2. فحص RAM
# =============================================
import psutil, shutil
ram = psutil.virtual_memory()
disk = shutil.disk_usage(".")
print(f"\n💾 RAM: {ram.available/1024**3:.1f}GB متاح / {ram.total/1024**3:.1f}GB إجمالي")
print(f"💽 Disk: {disk.free/1024**3:.1f}GB متاح")

if ram.available < 6 * 1024**3:
    print("⚠️  RAM أقل من 6GB — ممكن يكون بطيء، جرب أغلق البرامج الأخرى")

# =============================================
# 3. تحميل النموذج
# =============================================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"\n⏳ تحميل {MODEL} (مرة واحدة ~3GB)...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✅ تم في {time.time()-t0:.0f}s")

# =============================================
# 4. توليد 3 محادثات اختبار
# =============================================
TOPICS_TEST = [
    "casual greeting and small talk",
    "customer asking about return policy",
    "user needs help choosing a laptop",
]

SYSTEM_GEN = """You are a data generator. Generate a realistic chatbot conversation.
Output ONLY valid JSON, no markdown, no extra text.
Format: {"topic": "...", "conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
The assistant must sound natural, never robotic, never say 'As an AI'."""

SYSTEM_TRAIN = "You are a helpful, friendly assistant. You engage naturally, remember context, and adapt to any topic."

def generate(messages, max_tokens=400):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()

def to_chatml(conv):
    lines = [f"<|im_start|>system\n{SYSTEM_TRAIN}<|im_end|>"]
    for t in conv["conversation"]:
        lines.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    return "\n".join(lines)

print("\n🧪 توليد 3 محادثات اختبار...\n")
results = []
ok = 0

for i, topic in enumerate(TOPICS_TEST):
    print(f"[{i+1}/3] {topic}...", end=" ", flush=True)
    t0 = time.time()
    
    raw = generate([
        {"role": "system", "content": SYSTEM_GEN},
        {"role": "user",   "content": f'Generate a conversation about: "{topic}"'},
    ])
    
    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        data = json.loads(raw[s:e])
        assert "conversation" in data and len(data["conversation"]) >= 2
        results.append({"text": to_chatml(data), "topic": topic})
        ok += 1
        print(f"✅ ({time.time()-t0:.1f}s)")
        # اطبع مثال
        print(f"   👤 User: {data['conversation'][0]['content'][:70]}...")
        print(f"   🤖 Bot:  {data['conversation'][1]['content'][:70]}...")
    except Exception as ex:
        print(f"❌ ({ex})")

# =============================================
# 5. النتيجة
# =============================================
print(f"\n{'='*50}")
print(f"النتيجة: {ok}/3 محادثات ✅")

if ok >= 2:
    # احفظ كاختبار
    with open("test_output.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("💾 حُفظت في test_output.jsonl")
    print("\n🚀 كل شي شغّال! شغّل الـ pipeline الكامل:")
    print("   python3 generate_with_qwen.py")
else:
    print("\n⚠️  مشكلة في التوليد — راجع:")
    print("   1. RAM كافية؟ (خصك 8GB+)")
    print("   2. النموذج حُمّل صح؟")
    print("   3. شغّل: pip install transformers torch --upgrade")
