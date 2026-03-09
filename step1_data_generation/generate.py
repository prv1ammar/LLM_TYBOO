"""
RAVEN - Step 1: Multilingual Dataset Generation
=================================================
Génère un large dataset de conversations (AR / Darija / FR / EN)
avec labels de répétition pour fine-tuner Qwen 0.5B et 1.5B.

Output:
  data/train.jsonl   (~160k samples)
  data/val.jsonl     (~20k samples)
  data/test.jsonl    (~20k samples)
  data/stats.json
"""

import json
import random
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
from tqdm import tqdm

random.seed(42)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLES_PER_LANG = 50_000   # → 200k total


# ══════════════════════════════════════════════════════
#  TEMPLATES PAR LANGUE
# ══════════════════════════════════════════════════════

TEMPLATES = {
    "arabic": {
        "topics": [
            "الفيزياء الكمية", "برمجة بايثون", "تاريخ الأندلس",
            "كيفية كتابة مقال", "الذكاء الاصطناعي", "الطبخ الصحي", "الفلسفة", "الرياضيات",
        ],
        "repetitive": [
            "كيف يمكنني مساعدتك؟",
            "هل يمكنني مساعدتك في شيء آخر؟",
            "لا أفهم سؤالك، هل يمكنك إعادة الصياغة؟",
            "أنا هنا لمساعدتك.",
            "يسعدني مساعدتك.",
        ],
        "good_prefixes": [
            "بناءً على ما ذكرته سابقاً،",
            "إضافةً لما شرحته،",
            "للتوضيح أكثر،",
            "معلومة إضافية مهمة:",
        ],
        "user_questions": [
            "أريد الاستفسار عن {topic}",
            "ما هي إجراءات {topic}؟",
            "كيف يمكنني إتمام {topic}؟",
            "هل يمكنك مساعدتي في {topic}؟",
            "أحتاج معلومات عن {topic}",
        ],
    },
    "darija": {
        "topics": [
            "tbib dyal snan", "histoire dyal maroc", "kifash nt3lm python",
            "lyouma f marrakech", "رياضة", "kifash ndir tajine", " الذكاء الاصطناعي",
        ],
        "repetitive": [
            "kifash n3awnek?",
            "wash kayn shi haja khra?",
            "ma fhemtch, 3awd 3liha",
            "ana hna bash n3awnek",
            "mrhba, kifash nkhdmek?",
        ],
        "good_prefixes": [
            "zid 3la had l ma3louma:",
            "b7al ma golt lik, walakin zid:",
            "hadi ma3louma muhimma:",
            "b nisbat l so2al dyalek:",
        ],
        "user_questions": [
            "bghit n3ref 3la {topic}",
            "kifash ndayer {topic}?",
            "3tini ma3loumat 3la {topic}",
            "wash ymken t3awni f {topic}?",
            "nkhdm {topic} kifash?",
        ],
    },
    "french": {
        "topics": [
            "la physique quantique", "la révolution française", "les trous noirs",
            "l'apprentissage du machine learning", "les recettes végétariennes", "l'astronomie",
        ],
        "repetitive": [
            "Comment puis-je vous aider ?",
            "Puis-je faire autre chose pour vous ?",
            "Je ne comprends pas votre question.",
            "Je suis là pour vous aider.",
            "N'hésitez pas à me poser vos questions.",
        ],
        "good_prefixes": [
            "En complément de ma réponse précédente,",
            "Pour approfondir ce point,",
            "Voici une information supplémentaire :",
            "À noter également,",
        ],
        "user_questions": [
            "Je souhaite des informations sur {topic}",
            "Pouvez-vous m'expliquer {topic} ?",
            "Comment procéder pour {topic} ?",
            "J'ai une question concernant {topic}",
            "Aidez-moi avec {topic} s'il vous plaît",
        ],
    },
    "english": {
        "topics": [
            "quantum mechanics", "world war 2", "how to write a novel",
            "learning javascript", "healthy diets", "the stock market", "artificial intelligence",
        ],
        "repetitive": [
            "How can I help you?",
            "Is there anything else I can do for you?",
            "I don't understand your question.",
            "I'm here to assist you.",
            "Feel free to ask me anything.",
        ],
        "good_prefixes": [
            "Building on what I mentioned earlier,",
            "To add to my previous response,",
            "Here's an important additional point:",
            "Furthermore,",
        ],
        "user_questions": [
            "I'd like information about {topic}",
            "Can you help me with {topic}?",
            "How do I proceed with {topic}?",
            "I have a question about {topic}",
            "Please assist me with {topic}",
        ],
    },
}

# ══════════════════════════════════════════════════════
#  FAKE PII (pour test guardrails — Step 3)
# ══════════════════════════════════════════════════════

FAKE_PII_POOL = {
    "credit_card": [
        "4532 1234 5678 9010",
        "5123 4567 8901 2345",
    ],
    "id": ["AB123456", "CD987654", "HH112233"],
    "phone": ["0661234567", "0712345678", "+212661234567"],
    "email": ["client@example.ma", "user.test@example.com"],
    "name": ["John Doe", "Jane Smith", "Ali Riad"],
}


def inject_pii(text: str, prob: float = 0.12) -> Tuple[str, List[str]]:
    """Inject random fake PII into text with given probability."""
    if random.random() > prob:
        return text, []
    pii_type = random.choice(list(FAKE_PII_POOL.keys()))
    value = random.choice(FAKE_PII_POOL[pii_type])
    suffix = f" [{value}]"
    return text + suffix, [pii_type]


# ══════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════

@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    lang: str
    has_pii: bool = False
    pii_types: List[str] = field(default_factory=list)


@dataclass
class Sample:
    id: str
    lang: str
    turns: List[Turn]
    is_repetitive: bool         # label principal
    repetition_score: float     # 0.0 → 1.0
    has_pii: bool
    split: str                  # train / val / test

    def to_dict(self):
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════
#  GENERATION LOGIC
# ══════════════════════════════════════════════════════

def repetition_score(responses: List[str]) -> float:
    if len(responses) < 2:
        return 0.0
    hashes = [hashlib.md5(r.lower().strip().encode()).hexdigest() for r in responses]
    return 1.0 - len(set(hashes)) / len(hashes)


def make_repetitive_conv(lang: str, n_turns: int) -> List[Turn]:
    t = TEMPLATES[lang]
    turns = []
    repeated = random.choice(t["repetitive"])

    for i in range(n_turns):
        topic = random.choice(t["topics"])
        uq = random.choice(t["user_questions"]).format(topic=topic)
        uq, pii_u = inject_pii(uq)
        turns.append(Turn(role="user", content=uq, lang=lang,
                          has_pii=bool(pii_u), pii_types=pii_u))

        # Assistant repeats avec petites variations
        variation = repeated if random.random() < 0.90 else repeated + " ..."
        variation, pii_a = inject_pii(variation)
        turns.append(Turn(role="assistant", content=variation, lang=lang,
                          has_pii=bool(pii_a), pii_types=pii_a))
    return turns


def make_good_conv(lang: str, n_turns: int) -> List[Turn]:
    t = TEMPLATES[lang]
    turns = []

    for i in range(n_turns):
        topic = random.choice(t["topics"])
        uq = random.choice(t["user_questions"]).format(topic=topic)
        uq, pii_u = inject_pii(uq)
        turns.append(Turn(role="user", content=uq, lang=lang,
                          has_pii=bool(pii_u), pii_types=pii_u))

        prefix = random.choice(t["good_prefixes"])
        new_content = f"detail #{i+1} about '{topic}' — unique info {random.randint(1000,9999)}"
        resp = f"{prefix} {new_content}"
        resp, pii_a = inject_pii(resp)
        turns.append(Turn(role="assistant", content=resp, lang=lang,
                          has_pii=bool(pii_a), pii_types=pii_a))
    return turns


# ══════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════

def generate_dataset() -> List[Sample]:
    dataset = []
    stats = {lang: {"total": 0, "repetitive": 0, "pii": 0} for lang in TEMPLATES}

    for lang in TEMPLATES:
        print(f"\n[Step1] Generating {SAMPLES_PER_LANG:,} samples — lang={lang}")
        for i in tqdm(range(SAMPLES_PER_LANG)):
            is_rep = random.random() < 0.4
            n_turns = random.randint(3, 10)

            turns = make_repetitive_conv(lang, n_turns) if is_rep else make_good_conv(lang, n_turns)
            assistant_msgs = [t.content for t in turns if t.role == "assistant"]
            rep_sc = repetition_score(assistant_msgs)
            has_pii = any(t.has_pii for t in turns)

            # Split
            r = i / SAMPLES_PER_LANG
            split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")

            sample = Sample(
                id=f"{lang}_{i:06d}",
                lang=lang,
                turns=turns,
                is_repetitive=is_rep,
                repetition_score=rep_sc,
                has_pii=has_pii,
                split=split,
            )
            dataset.append(sample)

            stats[lang]["total"] += 1
            if is_rep:
                stats[lang]["repetitive"] += 1
            if has_pii:
                stats[lang]["pii"] += 1

    return dataset, stats


def save_dataset(dataset: List[Sample], stats: dict):
    for split in ["train", "val", "test"]:
        path = OUTPUT_DIR / f"{split}.jsonl"
        samples = [s for s in dataset if s.split == split]
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        print(f"[Step1] ✅ {split}.jsonl → {len(samples):,} samples")

    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Step1] ✅ stats.json saved")


if __name__ == "__main__":
    dataset, stats = generate_dataset()
    save_dataset(dataset, stats)
    print(f"\n[Step1] 🎉 Total: {len(dataset):,} samples generated!")
    for lang, s in stats.items():
        print(f"  {lang}: {s['total']:,} total | {s['repetitive']:,} repetitive | {s['pii']:,} with PII")
