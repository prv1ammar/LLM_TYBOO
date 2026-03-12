"""
RAVEN — High-Quality General Data Generator
===========================================
Generates diverse, high-quality synthetic data for a smart local assistant.
Focuses on greetings, on-hold messages, identity, and general FAQ.
Avoids repetition by using a combinatorial/variational approach.
"""

import os, json, random, argparse
from pathlib import Path

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

APPROX_BYTES_SFT = 400
DPO_PAIRS_PER_INTENT = 5000

# ══════════════════════════════════════════════════════════════
#  IDENTITY & SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════

SYSTEM = {
    "ar":     "أنت الساعد الذكي RAVEN. مساعدك الشخصي للإجابة على الأسئلة العامة والتحية.",
    "darija": "nta RAVEN, mssa3d daki. kanjawb 3la l'as2ila l3amma o kola nhar.",
    "fr":     "Vous êtes RAVEN, un assistant virtuel intelligent local. Répondez aux salutations et aux questions générales.",
    "en":     "You are RAVEN, an intelligent local assistant. Answer general questions and greetings.",
}

NAME = "RAVEN"

# ══════════════════════════════════════════════════════════════
#  VARIATIONAL FIELDS (Used to build unique rows)
# ══════════════════════════════════════════════════════════════

S_HI = {
    "en": ["Hi", "Hello", "Hey", "Greetings", "Good morning", "Good afternoon", "Hello there"],
    "fr": ["Bonjour", "Salut", "Coucou", "Bonsoir", "Allô", "Salutations"],
    "ar": ["مرحباً", "أهلاً", "أهلاً وسهلاً", "صباح الخير", "مساء الخير", "السلام عليكم"],
    "darija": ["Salam", "Ahlan", "Salam alaikom", "Sbh lkheir", "Msh lkheir", "Labas"]
}

S_ASST = {
    "en": ["I am RAVEN", "My name is RAVEN", "This is RAVEN here", "You are talking to RAVEN"],
    "fr": ["Je suis RAVEN", "Mon nom est RAVEN", "C'est RAVEN à votre service", "Vous parlez à RAVEN"],
    "ar": ["أنا RAVEN", "اسمي هو RAVEN", "معك المساعد RAVEN", "تتحدث الآن مع RAVEN"],
    "darija": ["Ana smiti RAVEN", "Smiti RAVEN", "Ma3ak RAVEN", "Ra katchouf m3a RAVEN"]
}

S_WAIT = {
    "en": ["Please wait a moment while I look that up.", "One second, I'm checking that for you.", "Hold on, I am processing your request.", "Just a moment, let me find that information."],
    "fr": ["Un instant s'il vous plaît, je vérifie.", "Attendez un moment, je traite votre demande.", "Un petit moment, je cherche l'information.", "Laissez-moi un instant pour trouver cela."],
    "ar": ["لحظة من فضلك، جاري البحث.", "انتظر ثانية، أنا أتحقق من ذلك.", "من فضلك انتظر بينما أقوم بمعالجة طلبك.", "لحظة واحدة، دعني أجد تلك المعلومات."],
    "darija": ["Tsna chwiya, kanchouf dikchi.", "Wahid taniya, rani kanchouf lk.", "Sf tsna taniya, kankhedam 3la l-tlaba dyalk.", "Blati wahed chwiya, bach n9aleb 3la dik l-ma3louma."]
}

# General Moroccan FAQ Data
MOROCCO_DATA = {
    "capitals": {"en": "Rabat", "fr": "Rabat", "ar": "الرباط", "darija": "Rabat"},
    "location": {"en": "North Africa", "fr": "Afrique du Nord", "ar": "شمال أفريقيا", "darija": "Chimal Afriqya"},
    "currency": {"en": "Moroccan Dirham (MAD)", "fr": "Dirham Marocain (MAD)", "ar": "الدرهم المغربي", "darija": "Dirham"}
}

# ══════════════════════════════════════════════════════════════
#  DYNAMIC GENERATORS
# ══════════════════════════════════════════════════════════════

def get_greeting(lang):
    h = random.choice(S_HI[lang])
    return f"{h}!" if random.random() < 0.5 else h

def get_identity(lang):
    i = random.choice(S_ASST[lang])
    tag = " your local assistant" if lang == "en" else " mssa3dk daki" if lang == "darija" else ""
    return f"{i}{tag}."

def make_variant_sft(lang):
    r = random.random()
    if r < 0.25: # Identity Call
        q = random.choice(["Who are you?", "What is your name?", "Tell me about yourself", "Identify yourself"]) if lang == "en" else \
            random.choice(["Chkoun nta?", "Smitk?", "Chnou nta?", "3rf rassek"]) if lang == "darija" else \
            random.choice(["Qui es-tu ?", "Comment t'appelles-tu ?", "C'est quoi ton nom ?", "Présente-toi"]) if lang == "fr" else \
            random.choice(["من أنت؟", "ما اسمك؟", "عرف بنفسك", "من تكون؟"])
        a = get_identity(lang)
    elif r < 0.50: # Greeting & Help
        q = get_greeting(lang)
        a = f"{get_greeting(lang)} {get_identity(lang)} How can I help you today?" if lang == "en" else \
            f"{get_greeting(lang)} {get_identity(lang)} Bach n9der n3awnk lyoum?" if lang == "darija" else \
            f"{get_greeting(lang)} {get_identity(lang)} Comment puis-je vous aider ?" if lang == "fr" else \
            f"{get_greeting(lang)} {get_identity(lang)} كيف يمكنني مساعدتك اليوم؟"
    elif r < 0.75: # Waiting / On-Hold
        q = random.choice(["Are you there?", "Wait", "Hold on", "Are you working on it?"]) if lang == "en" else \
            random.choice(["Wash nta hna?", "Tsna", "Blati", "Wash katsaybha?"]) if lang == "darija" else \
            random.choice(["Tu es là ?", "Attends", "Un instant", "Tu travailles dessus ?"]) if lang == "fr" else \
            random.choice(["هل أنت هنا؟", "انتظر", "لحظة", "هل تعمل عليها؟"])
        a = random.choice(S_WAIT[lang])
    else: # Morocco FAQ
        topic = random.choice(["capital", "location", "currency"])
        if topic == "capital":
            q = "What is the capital of Morocco?" if lang == "en" else "Chnou hiya 3asimat l-maghrib?" if lang == "darija" else "Capitale du Maroc ?" if lang == "fr" else "ما هي عاصمة المغرب؟"
            a = f"{MOROCCO_DATA['capitals'][lang]} is the capital." if lang == "en" else f"{MOROCCO_DATA['capitals'][lang]} hiya l-3asima." if lang == "darija" else f"C'est {MOROCCO_DATA['capitals'][lang]}." if lang == "fr" else f"العاصمة هي {MOROCCO_DATA['capitals'][lang]}."
        elif topic == "location":
            q = "Where is Morocco?" if lang == "en" else "Fin ja l-maghrib?" if lang == "darija" else "Où est le Maroc ?" if lang == "fr" else "أين يقع المغرب؟"
            a = f"It is in {MOROCCO_DATA['location'][lang]}." if lang == "en" else f"Ja f {MOROCCO_DATA['location'][lang]}." if lang == "darija" else f"En {MOROCCO_DATA['location'][lang]}." if lang == "fr" else f"يقع في {MOROCCO_DATA['location'][lang]}."
        else:
            q = "What is the currency of Morocco?" if lang == "en" else "Chnou hiya l-monnai dyal l-maghrib?" if lang == "darija" else "Monnaie du Maroc ?" if lang == "fr" else "ما هي عملة المغرب؟"
            a = MOROCCO_DATA['currency'][lang]

    convs = [
        {"from": "system", "value": SYSTEM[lang]},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a}
    ]
    return {"conversations": convs}

def make_variant_dpo(lang):
    # DPO is about preferring polite identity over "I don't know" or "I am a bank"
    q = "Who are you?" if lang == "en" else "Chkoun nta?" if lang == "darija" else "Qui es-tu ?" if lang == "fr" else "من أنت؟"
    chosen = get_identity(lang)
    rejected = "I am a banking assistant." if lang == "en" else "Ana mssa3d banki." if lang == "darija" else "Je suis un assistant bancaire." if lang == "fr" else "أنا مساعد بنكي."
    return {"prompt": q, "chosen": chosen, "rejected": rejected}

# ══════════════════════════════════════════════════════════════
#  CORE IO
# ══════════════════════════════════════════════════════════════

def gen_sft(lang, n, out_path, preview=False):
    if preview:
        print(f"\n[Preview SFT {lang}]")
        for _ in range(5): print(json.dumps(make_variant_sft(lang), indent=2, ensure_ascii=False))
        return 0
    
    print(f"[SFT] {lang.upper()} → {out_path.name} ({n:,})")
    wb = 0; total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            line = json.dumps(make_variant_sft(lang), ensure_ascii=False)
            f.write(line + "\n")
            wb += len(line.encode()) + 1
            total += 1
            if i % 100000 == 0 and i > 0: print(f"   {total:,}/{n:,}  {wb/1e9:.2f} GB")
    print(f"  ✅ {wb/1e9:.3f} GB  {total:,} samples")
    return wb

def gen_dpo(n, out_path, preview=False):
    if preview:
        print(f"\n[Preview DPO]")
        for _ in range(5): print(json.dumps(make_variant_dpo("en"), indent=2, ensure_ascii=False))
        return 0
    
    print(f"[DPO] → {out_path.name} ({n:,})")
    wb = 0; total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            lang = random.choice(["ar", "darija", "fr", "en"])
            line = json.dumps(make_variant_dpo(lang), ensure_ascii=False)
            f.write(line + "\n")
            wb += len(line.encode()) + 1
            total += 1
    print(f"  ✅ {total:,} DPO pairs  {wb/1e6:.1f} MB")
    return wb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang",     default="all", choices=["all","ar","darija","fr","en"])
    p.add_argument("--size",     type=float, default=1.0) # Size in GB total
    p.add_argument("--preview",  action="store_true")
    args = p.parse_args()
    
    langs = ["ar", "darija", "fr", "en"] if args.lang == "all" else [args.lang]
    n_sft = int((args.size * 1e9) / (len(langs) * APPROX_BYTES_SFT))
    
    if args.preview:
        gen_sft(langs[0], 0, OUT_DIR/"_", preview=True)
        gen_dpo(0, OUT_DIR/"_", preview=True)
        return

    total_wb = 0
    for lang in langs:
        total_wb += gen_sft(lang, n_sft, OUT_DIR/f"sft_{lang}.jsonl")
    
    total_wb += gen_dpo(DPO_PAIRS_PER_INTENT * 4, OUT_DIR/"dpo_all.jsonl")
    
    print(f"\n✅ SUCCESS: Generated {total_wb/1e9:.2f} GB of Variational FAQ data.")

if __name__ == "__main__":
    main()
