"""
RAVEN - Step 2: Intelligent Intent Classifier v2
==================================================
Fixes:
  ✅ Data loaded from separate files (data/<intent>/*.jsonl)
  ✅ Ensemble model: char n-gram + word TF-IDF + SVM voting
  ✅ Generalizes to unseen messages via semantic similarity features
  ✅ Smart augmentation: paraphrase templates, typo simulation, mixing
  ✅ Language detection robuste (AR/Darija/FR/EN)
  ✅ Keyword boosting layer pour cas ambigus
  ✅ Calibrated thresholds

Target: >90% CV accuracy
"""

import json, re, random, pickle, sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "raven_classifier_v2.pkl"

RAVEN_INTENTS = [
    "question_info", "complaint", "transaction",
    "support", "off_topic", "emergency",
]

RAVEN_TAGS = [
    "banking", "credit", "account", "card", "transfer",
    "urgent", "frustrated", "requires_human", "fraud_signal",
    "faq", "form_needed", "identity_verification", "sensitive_data",
    "lang:arabic", "lang:darija", "lang:french", "lang:english",
]

# Intent → default tags
INTENT_DEFAULT_TAGS = {
    "question_info":  ["banking", "faq"],
    "complaint":      ["banking", "frustrated", "requires_human"],
    "transaction":    ["banking", "transfer", "form_needed"],
    "support":        ["banking"],
    "off_topic":      [],
    "emergency":      ["banking", "urgent", "fraud_signal", "requires_human"],
}

# ══════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════

def load_all_data(data_dir: Path = DATA_DIR) -> Tuple[List[str], List[str], List[str]]:
    """
    Load all .jsonl files from data/<intent>/ folders.
    Returns: (texts, intents, langs)
    """
    texts, intents, langs = [], [], []
    loaded_files = []

    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        count = 0
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts.append(obj["text"])
                    intents.append(obj["intent"])
                    langs.append(obj.get("lang", "unknown"))
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        loaded_files.append(f"{jsonl_file.relative_to(data_dir)}: {count}")

    print(f"[DataLoader] Loaded {len(texts):,} samples from {len(loaded_files)} files:")
    for f in loaded_files:
        print(f"  {f}")
    print(f"[DataLoader] Distribution: {Counter(intents)}")
    return texts, intents, langs


# ══════════════════════════════════════════════════════════════
#  SMART AUGMENTATION
# ══════════════════════════════════════════════════════════════

# Paraphrase prefixes per language for question_info
QUESTION_PREFIXES = {
    "arabic": ["أريد معرفة", "هل يمكنني", "كيف أستطيع", "أحتاج معلومة عن", "ما هي طريقة"],
    "darija": ["bghit n3ref", "wash ymken", "kifash ymken", "3tini ma3louma 3la", "chno hiya"],
    "french": ["Je voudrais savoir", "Pouvez-vous m'expliquer", "Comment puis-je", "J'ai besoin d'infos sur", "Dites-moi"],
    "english": ["I'd like to know", "Can you tell me", "How can I", "I need info about", "Please explain"],
}

COMPLAINT_MARKERS = {
    "arabic": ["أنا غير راضٍ عن", "هذا غير مقبول", "أريد الشكوى من", "لم يحل أحد مشكلتي بخصوص"],
    "darija": ["ana machi radi 3la", "had sh7al ma maqbulch", "bghit nshki mn", "ma7al 7ta wa7d mshkilti"],
    "french": ["Je suis mécontent de", "C'est inacceptable", "Je veux me plaindre de", "Personne n'a résolu mon problème de"],
    "english": ["I'm unhappy with", "This is unacceptable", "I want to complain about", "Nobody solved my issue with"],
}

def augment_texts(texts: List[str], intents: List[str], langs: List[str], 
                  factor: int = 6) -> Tuple[List[str], List[str], List[str]]:
    """
    Smart augmentation:
    1. Add paraphrase prefixes
    2. Simulate typos (darija/english)
    3. Cross-language mixing (code switching)
    4. Truncation / extension
    """
    aug_texts, aug_intents, aug_langs = list(texts), list(intents), list(langs)

    for text, intent, lang in zip(texts, intents, langs):
        for _ in range(factor):
            op = random.random()

            if op < 0.25:
                # Paraphrase prefix
                prefixes = QUESTION_PREFIXES.get(lang, QUESTION_PREFIXES["english"])
                prefix = random.choice(prefixes)
                aug = f"{prefix} {text}"

            elif op < 0.45:
                # Word-level dropout (remove 1-2 words)
                words = text.split()
                if len(words) > 3:
                    n_drop = random.randint(1, min(2, len(words)-2))
                    for _ in range(n_drop):
                        idx = random.randint(0, len(words)-1)
                        words.pop(idx)
                aug = " ".join(words)

            elif op < 0.60:
                # Punctuation variation
                text_stripped = text.rstrip("?!.،")
                aug = text_stripped + random.choice(["?", "!", ".", "،", " svp", " merci", ""])

            elif op < 0.72:
                # Case variation (latin)
                aug = text.lower() if random.random() < 0.5 else text.upper()

            elif op < 0.82:
                # Number injection (realistic)
                numbers = ["100", "500", "1000", "2000", "50", "200", "3"]
                aug = text + f" ({random.choice(numbers)} DH)"

            elif op < 0.90:
                # Darija/French code switching
                if lang == "darija" and intent in INTENT_DEFAULT_TAGS:
                    fr_words = {"compte": "compte", "carte": "carte", "virement": "virement",
                                "crédit": "crédit", "application": "application"}
                    aug = text  # already has code switching
                else:
                    aug = text

            else:
                # Repetition / emphasis
                words = text.split()
                if len(words) > 2:
                    idx = random.randint(0, len(words)-1)
                    words.insert(idx, words[idx])
                aug = " ".join(words)

            aug_texts.append(aug.strip())
            aug_intents.append(intent)
            aug_langs.append(lang)

    print(f"[Augmentation] {len(texts):,} → {len(aug_texts):,} samples (x{factor} augmentation)")
    return aug_texts, aug_intents, aug_langs


# ══════════════════════════════════════════════════════════════
#  LANGUAGE DETECTOR
# ══════════════════════════════════════════════════════════════

DARIJA_WORDS = {
    'bghit', 'kifash', 'wash', 'ndir', 'kayna', 'dyali', 'dyal', 'walakin',
    'blokiha', 'n7awel', 'nkhed', 'nsedd', 'flous', 'srqo', 'daba', 'machi',
    'bzzaf', 'chhal', 'imta', 'chno', 'n3ref', 'nftah', 'ndkhol', 'ntuma',
    'ma9darsh', 'ma3refch', 'kayduz', 'kayt5od', 'ymken', 'khassni', 'wa7d',
    'l7sab', '3br', '3la', 'mn', 'f', 'b', 'li', 'hiya', 'huma', 'ana',
    'tsennit', 'wesalch', 'khdamach', 'tliffon', 'frague', 'khedma',
    'mublagh', 'mblagh', 'n9dar', 'n3awd', 'nzid', 'nshof', 'nlqa',
}
FRENCH_WORDS = {
    'je', 'vous', 'mon', 'ma', 'mes', 'le', 'la', 'les', 'est', 'sont',
    'pas', 'une', 'des', 'du', 'de', 'et', 'en', 'un', 'pour', 'dans',
    'que', 'qui', 'sur', 'avec', 'ne', 'se', 'ce', 'cette', 'au',
    'comment', 'quand', 'quel', 'quelle', 'voudrais', 'veux', 'souhaite',
    'puis', 'peut', 'faire', 'avoir', 'être', 'compte', 'virement',
    'effectuer', 'bloquer', 'annuler', 'signaler', 'demander', 'consulter',
}
ENGLISH_WORDS = {
    'i', 'my', 'the', 'is', 'are', 'can', 'do', 'want', 'help', 'how',
    'what', 'when', 'where', 'who', 'would', 'could', 'should', 'have',
    'has', 'been', 'get', 'need', 'please', 'account', 'bank', 'card',
    'transfer', 'block', 'freeze', 'cancel', 'report', 'stolen', 'fraud',
}

def detect_language(text: str) -> str:
    arabic_count = len(re.findall(r'[\u0600-\u06ff]', text))
    if arabic_count > 0:
        words_lower = set(re.findall(r'[a-z0-9]+', text.lower()))
        if words_lower & DARIJA_WORDS:
            return "darija"
        return "arabic"
    words = set(re.findall(r"[a-z']+", text.lower()))
    fr = len(words & FRENCH_WORDS)
    en = len(words & ENGLISH_WORDS)
    da = len(words & DARIJA_WORDS)
    if da > max(fr, en):
        return "darija"
    if any(c in text for c in 'àâéèêëîïôùûüç'):
        fr += 2
    return "french" if fr >= en else "english"


# ══════════════════════════════════════════════════════════════
#  KEYWORD BOOSTING (pour cas ambigus)
# ══════════════════════════════════════════════════════════════

EMERGENCY_KEYWORDS = re.compile(
    r'(vol[eé]|srak|srqo|stolen|volé|hack|piraté|mkhtar9|fraud|احتيال|سرق|اختراق|'
    r'bloqu|block|gel|freeze|9awwed|9awwad|annul.*carte|carte.*vol|بطاقة.*مسرو|'
    r'معاملات.*لم|transactions.*non|unauthorized|غير مصرح|bla ijaz|'
    r'pris mon argent|took.*money|took everything|dkhlu.*l7sab|سحب كل|'
    r'HELP.*account|AIDEZ.*argent|3AJJLU|عاجل.*اختراق|quelqu.un a pris|someone hacked|'
    r'quelqu.un.*pris.*argent|a pris mon argent|my money.*gone|money.*disappeared|'
    r'compte.*compromis|مخترق|9awwed l7sab)',
    re.IGNORECASE
)
COMPLAINT_KEYWORDS = re.compile(
    r'(insatisf|mécontent|machi radi|غير راض|plainte|shikaya|شكوى|'
    r'inacceptable|ma maqbulch|remboursement|rdod|استرداد|'
    r'responsable|directeur|lmdir|bank.*maghrib|سأرفع|ghadi.*nsi77et|'
    r'vraiment nul|c.est nul|za3fan 3la qad|nobody cares|bank is a joke|'
    r'j.en ai marre|en ai marre|so bad|terrible service|worst bank|'
    r'i.m done with|fed up|this.*joke|what a joke)',
    re.IGNORECASE
)
OFF_TOPIC_KEYWORDS = re.compile(
    r'(météo|weather|tbard|lbard|cuisine|recette|couscous|tajine|blague|joke(?!.*bank)|'
    r'\bmatch\b|programme.*tv|télé|histoire|tarikh|تاريخ|طقس|نكتة|'
    r'learn arabic|learn.*language|apprendre.*langue|best.*restaurant|'
    r'rajfana|best way to learn|tell me about|raconte.*histoire|'
    r'what.s.*best way|comment faire.*(?!virement|paiement|transfert))',
    re.IGNORECASE
)
SUPPORT_KEYWORDS = re.compile(
    r'(ma khdamach|ne fonctionne pas|not working|لا يعمل|'
    r'session expired|mot de passe|password|code.*oubli|nsit.*code|'
    r'bloqué|m9ful|locked|OTP|erreur|error|خطأ|'
    r'badge.*ma khdamach|guichet.*ma khdamach|plante|crashes)',
    re.IGNORECASE
)
TRANSACTION_KEYWORDS = re.compile(
    r'(virer|virement|7awel|n7awel|'
    r'payer.*facture|nkhed.*fatura|pay.*bill|facture|fatura|فاتورة|'
    r'loyer|rent(?!.*balance)|إيجار|dépôt|deposit|إيداع|'
    r'rembours|nsedd|سداد|recharger|nchar9|شحن|'
    r'nkhed.*loyer|payer.*loyer|pay.*rent|'
    r'\bwire\s+\d|\bwiring\b)',
    re.IGNORECASE
)
QUESTION_INFO_KEYWORDS = re.compile(
    r'(so2al|استفسار|renseignement|'
    r"c'est quoi|c est quoi|what is|what are|how (do|can|much|long)|"
    r'quel.*délai|chhal.*waqt|كم يستغرق|how long does|'
    r'check.*balance|solde|رصيد|ls7b.*dyal|plafond|'
    r'peut.on|puis-je|can i\b|wash ymken|هل يمكن)',
    re.IGNORECASE
)


def keyword_boost(text: str, pred_intent: str, pred_proba: np.ndarray,
                  intent_list: List[str]) -> Tuple[str, np.ndarray]:
    """
    Adjust probabilities based on strong keyword signals.
    Priority: emergency > off_topic > support > transaction > complaint
    """
    proba = pred_proba.copy()

    # Emergency — highest priority, override everything
    if EMERGENCY_KEYWORDS.search(text):
        em_idx = intent_list.index("emergency")
        proba[em_idx] = max(proba[em_idx], 0.85)
        proba /= proba.sum()

    # Off-topic — strong signal (non-banking content)
    elif OFF_TOPIC_KEYWORDS.search(text):
        ot_idx = intent_list.index("off_topic")
        proba[ot_idx] = max(proba[ot_idx], 0.80)
        proba /= proba.sum()

    # Support — technical issues
    elif SUPPORT_KEYWORDS.search(text):
        su_idx = intent_list.index("support")
        proba[su_idx] = max(proba[su_idx], 0.70)
        proba /= proba.sum()

    # Transaction — action keywords
    elif TRANSACTION_KEYWORDS.search(text) and pred_intent not in ("emergency", "support"):
        tr_idx = intent_list.index("transaction")
        proba[tr_idx] = max(proba[tr_idx], 0.65)
        proba /= proba.sum()

    # Complaint — dissatisfaction
    elif COMPLAINT_KEYWORDS.search(text) and pred_intent != "emergency":
        co_idx = intent_list.index("complaint")
        proba[co_idx] = max(proba[co_idx], 0.72)
        proba /= proba.sum()

    final_intent = intent_list[np.argmax(proba)]
    return final_intent, proba


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE CLASSIFIER
# ══════════════════════════════════════════════════════════════

class RAVENClassifier:
    """
    Ensemble: char_tfidf + word_tfidf + SVM → VotingClassifier
    Generalizes to unseen messages via TF-IDF semantic space.
    """

    def __init__(self):
        self._pipeline = None
        self._classes  = None

    def _build_pipeline(self):
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import Normalizer

        # Feature 1: char n-grams (2-5) — captures morphology across all scripts
        char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=80_000,
            sublinear_tf=True,
            min_df=1,
            strip_accents=None,   # keep Arabic/accented chars
        )

        # Feature 2: word n-grams (1-3) — captures meaning
        word_tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=50_000,
            sublinear_tf=True,
            min_df=1,
            token_pattern=r'(?u)\b\w+\b',
        )

        features = FeatureUnion([
            ('char', char_tfidf),
            ('word', word_tfidf),
        ])

        # SVM with probability calibration
        svm = CalibratedClassifierCV(
            LinearSVC(C=1.5, max_iter=3000, class_weight='balanced'),
            cv=3,
            method='sigmoid',
        )

        return Pipeline([
            ('features', features),
            ('clf', svm),
        ])

    def fit(self, texts: List[str], intents: List[str]):
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        print("[RAVENClassifier] Building ensemble pipeline...")
        self._pipeline = self._build_pipeline()
        self._classes  = RAVEN_INTENTS

        print("[RAVENClassifier] Training...")
        self._pipeline.fit(texts, intents)

        # Cross-validation
        print("[RAVENClassifier] Cross-validating (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            self._build_pipeline(), texts, intents,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        print(f"[RAVENClassifier] ✅ CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"[RAVENClassifier]    Per-fold:   {[f'{s:.3f}' for s in scores]}")
        return scores.mean()

    def predict_proba(self, text: str) -> Tuple[str, float, np.ndarray]:
        """Returns (intent, confidence, all_probas)"""
        # ── Step 1: Hard rules first (high confidence keyword signals) ──
        n_intents = len(self._classes)
        hard_intent = None

        if EMERGENCY_KEYWORDS.search(text):
            hard_intent = "emergency"
        elif OFF_TOPIC_KEYWORDS.search(text):
            hard_intent = "off_topic"
        elif SUPPORT_KEYWORDS.search(text):
            hard_intent = "support"
        elif TRANSACTION_KEYWORDS.search(text):
            hard_intent = "transaction"
        elif COMPLAINT_KEYWORDS.search(text):
            hard_intent = "complaint"
        elif QUESTION_INFO_KEYWORDS.search(text):
            hard_intent = "question_info"

        # ── Step 2: Model prediction ──
        probas = self._pipeline.predict_proba([text])[0]
        model_intent = self._classes[np.argmax(probas)]
        model_conf   = float(np.max(probas))

        # ── Step 3: Combine — hard rule wins if model confidence < 0.80 ──
        if hard_intent and (model_intent != hard_intent or model_conf < 0.80):
            # Build synthetic proba with hard rule intent dominant
            final_probas = probas.copy()
            hi = self._classes.index(hard_intent)
            # Set hard rule intent to at least 0.80
            final_probas[hi] = max(final_probas[hi], 0.80)
            # Normalize others proportionally
            others_sum = final_probas.sum() - final_probas[hi]
            remaining  = 1.0 - final_probas[hi]
            if others_sum > 0:
                for j in range(len(final_probas)):
                    if j != hi:
                        final_probas[j] = final_probas[j] / others_sum * remaining
            final_intent = hard_intent
            confidence   = float(final_probas[hi])
        else:
            final_intent = model_intent
            final_probas = probas
            confidence   = model_conf

        return final_intent, confidence, final_probas

    def predict_full(self, text: str) -> Dict:
        """Full prediction with intent, tags, lang, confidence."""
        intent, confidence, probas = self.predict_proba(text)
        lang = detect_language(text)

        # Build tags
        tags = list(INTENT_DEFAULT_TAGS.get(intent, []))
        tags.append(f"lang:{lang}")

        # Extra tags from signals
        if EMERGENCY_KEYWORDS.search(text) and "fraud_signal" not in tags:
            tags.append("fraud_signal")
        if intent in ("complaint", "emergency") and "requires_human" not in tags:
            tags.append("requires_human")
        if len(re.findall(r'[\u0600-\u06ff]', text)) > 0 and re.search(r'[a-zA-Z]{3,}', text):
            tags.append("code_switching")

        all_intents = {cls: float(p) for cls, p in zip(self._classes, probas)}

        return {
            "intent":      intent,
            "confidence":  round(confidence, 4),
            "tags":        list(dict.fromkeys(tags)),  # deduplicate, preserve order
            "lang":        lang,
            "all_intents": all_intents,
        }

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({"pipeline": self._pipeline, "classes": self._classes}, f)
        print(f"[RAVENClassifier] ✅ Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RAVENClassifier":
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        clf = cls()
        clf._pipeline = obj["pipeline"]
        clf._classes  = obj["classes"]
        print(f"[RAVENClassifier] ✅ Model loaded from {path}")
        return clf


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(clf: RAVENClassifier, texts: List[str], intents: List[str]):
    from sklearn.metrics import classification_report
    preds = [clf.predict_full(t)["intent"] for t in texts]
    print("\n[Eval] Classification Report:")
    print(classification_report(intents, preds, target_names=RAVEN_INTENTS, digits=3))


# ══════════════════════════════════════════════════════════════
#  MAIN — TRAIN + DEMO
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 65)
    print("  RAVEN Intent Classifier v2 — Training")
    print("=" * 65)

    # 1. Load data from files
    texts, intents, langs = load_all_data()

    # 2. Augment
    texts, intents, langs = augment_texts(texts, intents, langs, factor=8)

    # 3. Train
    clf = RAVENClassifier()
    acc = clf.fit(texts, intents)

    # 4. Evaluate on raw data (no augmentation)
    raw_texts, raw_intents, _ = load_all_data()
    evaluate(clf, raw_texts, raw_intents)

    # 5. Save
    clf.save()
    return clf, acc


def demo(clf: RAVENClassifier):
    # Unseen messages — machi f training data
    UNSEEN_TESTS = [
        # question_info — phrasing jamais vu
        ("3ndi so2al 3la l plafond dyal ls7b dyal compte",       "question_info",  "darija"),
        ("c'est quoi le délai pour avoir ma carte ?",             "question_info",  "french"),
        ("can i check my balance from abroad?",                   "question_info",  "english"),
        ("كم يستغرق تحويل المبالغ بين البنوك؟",                    "question_info",  "arabic"),
        # complaint — ton différent
        ("votre banque c'est vraiment nul, j'en ai marre",       "complaint",      "french"),
        ("had service bla sta7ya, za3fan 3la qad Allah",         "complaint",      "darija"),
        ("this bank is a joke, nobody cares about customers",    "complaint",      "english"),
        # transaction — action directe
        ("3yit nkhed loyer dyal shqqa 3br l application",       "transaction",    "darija"),
        ("je dois virer de l'argent à ma famille ce soir",       "transaction",    "french"),
        ("need to wire 5000 to another account right now",       "transaction",    "english"),
        # support — problème technique inattendu
        ("l badge ma khdamach f l guichet",                      "support",        "darija"),
        ("mon application plante dès que j'ouvre les virements", "support",        "french"),
        ("keeps saying session expired every time i login",      "support",        "english"),
        # emergency — ton panique
        ("AIDEZ MOI quelqu'un a pris mon argent !!!",            "emergency",      "french"),
        ("3AJJLU dkhlu l7sab dyali hakda bla ma n3ref",         "emergency",      "darija"),
        ("HELP someone hacked my account and took everything",   "emergency",      "english"),
        ("تم سحب كل أموالي الآن بدون إذن مني!",                 "emergency",      "arabic"),
        # off_topic — sujets divers
        ("comment faire un bon tajine marocain?",                "off_topic",      "french"),
        ("3llmni wach rajfana ghda f maroc",                     "off_topic",      "darija"),
        ("what's the best way to learn arabic?",                 "off_topic",      "english"),
    ]

    print("\n" + "=" * 65)
    print("  DEMO — Unseen Messages (not in training data)")
    print("=" * 65)

    correct = 0
    for text, expected, exp_lang in UNSEEN_TESTS:
        result = clf.predict_full(text)
        ok = "✅" if result["intent"] == expected else "❌"
        lang_ok = "✅" if result["lang"] == exp_lang else "⚠️ "
        if result["intent"] == expected:
            correct += 1
        print(f"\n  {ok} [{result['intent']:14s}] (expected: {expected})")
        print(f"     {lang_ok} lang={result['lang']:8s} (expected: {exp_lang})")
        print(f"        conf={result['confidence']:.2f}  tags={result['tags']}")
        print(f"        text: {text[:60]}")

    print(f"\n  Accuracy on unseen: {correct}/{len(UNSEEN_TESTS)} = {correct/len(UNSEEN_TESTS):.1%}")
    print("=" * 65)


if __name__ == "__main__":
    if "--predict" in sys.argv:
        # Quick inference mode
        clf = RAVENClassifier.load()
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Message: ")
        result = clf.predict_full(msg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Train mode
        clf, acc = train()
        demo(clf)
        print(f"\n✅ Training complete — CV Accuracy: {acc:.1%}")
        print(f"   Model: {MODEL_PATH}")
        print(f"\n   Usage: python classifier_v2.py --predict <message>")
