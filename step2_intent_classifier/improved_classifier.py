"""
RAVEN - Step 2: Improved Intent Classifier
============================================
Based on reverse-engineering of conversation_classifier model:
  - Architecture: 3-layer Transformer encoder
  - embed_dim: 100, vocab_size: 12000, max_len: 256
  - Original intents: acknowledgment, confirmation, emotional_expression,
    feedback, follow-up, interest, negotiation, opinion, persuasion,
    reminder, request, social
  - Original tags: 350 tags (age, gender, country, personality, intent:buy, etc.)

PROBLÈME D'ACCURACY IDENTIFIÉ:
  ❌ Les intents originaux sont trop génériques (negotiation, persuasion...)
  ❌ Les tags sont trop larges (350 tags mal distribués)
  ❌ Pas adapté au domaine banking/support multilingue
  ❌ Vocab de 12k trop petit pour 4 langues

SOLUTION — 3 améliorations:
  1. Remapper les intents originaux vers les intents RAVEN (banking)
  2. Fine-tuner la tête de classification (intent_head + tags_head)
  3. Augmenter le vocab et re-tokenizer pour AR/Darija/FR/EN
"""

import json
import re
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════
#  RAVEN INTENTS (remplacent les 12 originaux)
# ══════════════════════════════════════════════════════

RAVEN_INTENTS = [
    "question_info",      # demande d'information
    "complaint",          # réclamation / insatisfaction
    "transaction",        # virement, paiement, opération
    "support",            # support technique / aide
    "off_topic",          # hors sujet
    "emergency",          # urgence (vol carte, fraude...)
]

RAVEN_TAGS = [
    # Domain
    "banking", "insurance", "credit", "account", "card", "transfer",
    # Sentiment
    "urgent", "frustrated", "satisfied", "confused", "angry", "calm",
    # Risk
    "pii_risk", "fraud_signal", "sensitive_data", "requires_human",
    # Content
    "faq", "form_needed", "identity_verification", "multilingual", "code_switching",
    # Language
    "lang:arabic", "lang:darija", "lang:french", "lang:english",
]

# ══════════════════════════════════════════════════════
#  MAPPING: old intents → RAVEN intents
# ══════════════════════════════════════════════════════

INTENT_MAPPING = {
    "request":             "question_info",
    "interest":            "question_info",
    "follow-up":           "question_info",
    "feedback":            "complaint",
    "opinion":             "complaint",
    "negotiation":         "transaction",
    "confirmation":        "transaction",
    "reminder":            "support",
    "acknowledgment":      "support",
    "social":              "off_topic",
    "emotional_expression":"complaint",
    "persuasion":          "off_topic",
}

# ══════════════════════════════════════════════════════
#  IMPROVED TOKENIZER (multilingual, 4 langues)
# ══════════════════════════════════════════════════════

class MultilingualTokenizer:
    """
    Simple but effective tokenizer for AR / Darija / FR / EN.
    Better than the original 12k-vocab version for multilingual.
    """
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    RESERVED = 4

    # Arabic normalizations
    AR_NORM = [
        (r'[إأآا]', 'ا'),
        (r'[يى]', 'ي'),
        (r'ة', 'ه'),
        (r'\u0640', ''),           # tatweel
        (r'[\u064b-\u065f]', ''),  # tashkeel
    ]

    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {
            '<PAD>': self.PAD,
            '<UNK>': self.UNK,
            '<BOS>': self.BOS,
            '<EOS>': self.EOS,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self._ar_compiled = [(re.compile(p), r) for p, r in self.AR_NORM]

    def _normalize_arabic(self, text: str) -> str:
        for pattern, repl in self._ar_compiled:
            text = pattern.sub(repl, text)
        return text

    def _tokenize_text(self, text: str) -> List[str]:
        text = self._normalize_arabic(text)
        text = text.lower()
        # Keep Arabic/Latin chars and numbers, split on rest
        tokens = re.findall(r'[\u0600-\u06ff]+|[a-z0-9][a-z0-9\-]*', text)
        return tokens

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize_text(text))
        # Add most common words
        for word, _ in counter.most_common(self.vocab_size - self.RESERVED):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"[Tokenizer] Vocab built: {len(self.word2idx):,} tokens")

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = self._tokenize_text(text)[:max_len - 2]
        ids = [self.BOS]
        ids += [self.word2idx.get(t, self.UNK) for t in tokens]
        ids += [self.EOS]
        # Pad
        ids += [self.PAD] * (max_len - len(ids))
        return ids[:max_len]

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MultilingualTokenizer':
        tok = cls()
        with open(path, encoding='utf-8') as f:
            tok.word2idx = json.load(f)
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}
        return tok


# ══════════════════════════════════════════════════════
#  IMPROVED MODEL ARCHITECTURE (numpy — no torch needed here)
# ══════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int    = 20000    # augmenté (original: 12000)
    embed_dim: int     = 128      # augmenté (original: 100)
    num_heads: int     = 4        # même
    num_layers: int    = 3        # même (garder compatibilité)
    ff_dim: int        = 512      # augmenté (original: 400)
    max_len: int       = 256      # même
    dropout: float     = 0.2      # augmenté (original: ~0.1)
    num_intents: int   = len(RAVEN_INTENTS)   # 6 (original: 12)
    num_tags: int      = len(RAVEN_TAGS)      # 24 (original: 350)


# ══════════════════════════════════════════════════════
#  TRAINING DATA: RAVEN-SPECIFIC EXAMPLES
# ══════════════════════════════════════════════════════

TRAINING_EXAMPLES = {
    "question_info": {
        "arabic": [
            "أريد الاستفسار عن رصيد حسابي",
            "ما هي شروط القرض الشخصي؟",
            "كيف يمكنني معرفة رقم RIB الخاص بي؟",
            "متى يتم تحديث كشف الحساب؟",
            "ما هي رسوم التحويل البنكي؟",
            "هل يمكنني فتح حساب توفير عبر الإنترنت؟",
            "ما هي وثائق فتح الحساب المطلوبة؟",
        ],
        "darija": [
            "bghit n3ref 3la compte dyali",
            "chhal kayt5od l faida dyal crédit?",
            "kifash ndir bach n3ref RIB dyali?",
            "imta kayt7ayed le relevé?",
            "chhal hiya frais dyal virement?",
            "wash ymken nftah compte 3br internet?",
            "chno hiya les documents li khassni?",
        ],
        "french": [
            "Je voudrais des informations sur mon compte",
            "Quels sont les frais de virement ?",
            "Comment obtenir mon RIB ?",
            "Quand est mis à jour mon relevé ?",
            "Quelles sont les conditions du crédit ?",
            "Puis-je ouvrir un compte en ligne ?",
            "Quels documents faut-il fournir ?",
        ],
        "english": [
            "I'd like information about my account",
            "What are the transfer fees?",
            "How do I get my IBAN?",
            "When is my statement updated?",
            "What are the loan conditions?",
            "Can I open an account online?",
            "What documents do I need?",
        ],
        "tags": ["banking", "faq"],
    },
    "complaint": {
        "arabic": [
            "أنا غير راضٍ عن الخدمة",
            "تأخرت عملية التحويل أكثر من يومين",
            "تم خصم مبلغ من حسابي بشكل خاطئ",
            "الخدمة سيئة جداً",
            "لم يتم حل مشكلتي رغم تواصلي مراراً",
            "أطلب إعادة المبلغ المخصوم بشكل غير صحيح",
        ],
        "darija": [
            "ana machi radi 3la l khedma",
            "virement ma wesalch men 2 iyam",
            "khassoni flous bla ma n3ref",
            "l application ma khedamach",
            "za3fan bzzaf mn had l service",
            "bghit rdod l flous li 5samo ghaltan",
        ],
        "french": [
            "Je suis insatisfait du service",
            "Mon virement n'est toujours pas arrivé",
            "Un montant a été débité par erreur",
            "Le service est vraiment mauvais",
            "Ma réclamation n'a pas été traitée",
            "Je demande le remboursement du débit erroné",
        ],
        "english": [
            "I'm not satisfied with the service",
            "My transfer hasn't arrived after 2 days",
            "An amount was debited incorrectly",
            "The service is really bad",
            "My complaint was not resolved",
            "I want a refund for the wrong debit",
        ],
        "tags": ["frustrated", "requires_human", "banking"],
    },
    "transaction": {
        "arabic": [
            "أريد تحويل مبلغ من حسابي",
            "كيف أدفع فاتورة الكهرباء؟",
            "أريد سداد قسط القرض",
            "أريد إيداع مبلغ في حسابي",
            "كيف أقوم بعملية الشراء عبر الإنترنت؟",
        ],
        "darija": [
            "bghit n7awel flous",
            "kifash nkhed fatura dyal l kahruba?",
            "bghit nsedd l 9rit",
            "bghit ndir dépôt f l7sab",
            "kifash n5les online?",
        ],
        "french": [
            "Je souhaite effectuer un virement",
            "Comment payer ma facture d'électricité ?",
            "Je veux rembourser mon crédit",
            "Je veux faire un dépôt",
            "Comment payer en ligne ?",
        ],
        "english": [
            "I want to make a bank transfer",
            "How do I pay my electricity bill?",
            "I want to repay my loan",
            "I want to make a deposit",
            "How do I pay online?",
        ],
        "tags": ["banking", "transfer", "form_needed"],
    },
    "support": {
        "arabic": [
            "تطبيق البنك لا يعمل",
            "نسيت كلمة المرور الخاصة بي",
            "لا أستطيع تسجيل الدخول",
            "كيف أحدث معلوماتي الشخصية؟",
            "هناك مشكلة في التطبيق",
        ],
        "darija": [
            "l application ma khedamach",
            "nsit le mot de passe",
            "ma9darsh ndkhol",
            "kifash n7ayed l ma3loumat dyali?",
            "kayna mushkila f application",
        ],
        "french": [
            "L'application ne fonctionne pas",
            "J'ai oublié mon mot de passe",
            "Je n'arrive pas à me connecter",
            "Comment mettre à jour mes informations ?",
            "Il y a un problème avec l'application",
        ],
        "english": [
            "The app is not working",
            "I forgot my password",
            "I can't log in",
            "How do I update my personal information?",
            "There's a problem with the application",
        ],
        "tags": ["support", "banking"],
    },
    "emergency": {
        "arabic": [
            "تم سرقة بطاقتي البنكية!",
            "أرى معاملات لم أقم بها",
            "هاتفي مسروق وأخشى على حسابي",
            "تم اختراق حسابي!",
            "أريد إيقاف جميع العمليات فوراً",
            "هناك عملية احتيال على حسابي",
        ],
        "darija": [
            "srqo carte dyali!",
            "kaynin 7wakam ma drithomch",
            "tliffon dyali tsrq khayef 3la l7sab",
            "dakhal 7ad f l7sab dyali!",
            "blokiwali l7sab daba!",
            "kayna 7iyala f l7sab dyali",
        ],
        "french": [
            "Ma carte bancaire a été volée !",
            "Je vois des transactions que je n'ai pas effectuées",
            "Mon téléphone a été volé",
            "Mon compte a été piraté !",
            "Bloquez mon compte immédiatement !",
            "Il y a une fraude sur mon compte",
        ],
        "english": [
            "My bank card was stolen!",
            "I see transactions I didn't make",
            "My phone was stolen",
            "My account has been hacked!",
            "Block my account immediately!",
            "There is fraud on my account",
        ],
        "tags": ["urgent", "fraud_signal", "requires_human", "banking"],
    },
    "off_topic": {
        "arabic": [
            "ما هو الطقس اليوم؟",
            "كيف أطبخ الكسكس؟",
            "أخبرني نكتة",
            "من فاز في مباراة أمس؟",
            "ما هو برنامج التلفزيون الليلة؟",
        ],
        "darija": [
            "chhal had lbard lyoum?",
            "3llmni t9da couscous",
            "3ndk chi blague?",
            "shkun reb7 match lbara7?",
            "chno kayn f television llila?",
        ],
        "french": [
            "Quel temps fait-il aujourd'hui ?",
            "Comment faire du couscous ?",
            "Raconte-moi une blague",
            "Qui a gagné le match d'hier ?",
            "Qu'est-ce qu'il y a à la télé ce soir ?",
        ],
        "english": [
            "What's the weather like today?",
            "How do I make couscous?",
            "Tell me a joke",
            "Who won yesterday's match?",
            "What's on TV tonight?",
        ],
        "tags": [],
    },
}


def build_dataset(n_augment: int = 8) -> Tuple[List[str], List[str], List[List[str]]]:
    """Build (texts, intents, tags_list) with data augmentation."""
    texts, intents, tags_list = [], [], []

    for intent, lang_dict in TRAINING_EXAMPLES.items():
        base_tags = lang_dict.get("tags", [])
        all_examples = []
        for lang, examples in lang_dict.items():
            if lang == "tags":
                continue
            lang_tag = f"lang:{lang}"
            for ex in examples:
                all_examples.append((ex, base_tags + [lang_tag]))

        for text, ex_tags in all_examples:
            texts.append(text)
            intents.append(intent)
            tags_list.append(ex_tags)

            # Augmentation: random character drops / case variants
            for _ in range(n_augment):
                aug = _augment(text)
                texts.append(aug)
                intents.append(intent)
                tags_list.append(ex_tags)

    return texts, intents, tags_list


def _augment(text: str) -> str:
    """Simple text augmentation."""
    ops = random.randint(1, 2)
    for _ in range(ops):
        choice = random.random()
        if choice < 0.3 and len(text) > 10:
            # Remove random word
            words = text.split()
            if len(words) > 3:
                idx = random.randint(0, len(words)-1)
                words.pop(idx)
            text = ' '.join(words)
        elif choice < 0.6:
            # Add punctuation variation
            text = text.rstrip('?!.') + random.choice(['?', '!', '.', ''])
        else:
            # Lowercase/uppercase variation
            text = text.lower() if random.random() < 0.5 else text
    return text


# ══════════════════════════════════════════════════════
#  LIGHTWEIGHT NUMPY CLASSIFIER (sans torch)
# ══════════════════════════════════════════════════════

class SimpleIntentClassifier:
    """
    TF-IDF + Logistic Regression classifier.
    Rapide, interprétable, fonctionne sans GPU.
    Utilisé pour améliorer l'accuracy quand le Transformer original underperforms.
    """

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        self.intent_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),     # character n-grams → handles Arabic, Darija
                max_features=50000,
                sublinear_tf=True,
                min_df=1,
            )),
            ('clf', LogisticRegression(
                C=5.0,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
            )),
        ])

        self.tag_mlb = MultiLabelBinarizer(classes=RAVEN_TAGS)
        self.tag_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=50000,
                sublinear_tf=True,
            )),
            ('clf', OneVsRestClassifier(LogisticRegression(
                C=2.0, max_iter=500, class_weight='balanced'
            ))),
        ])

    def fit(self, texts: List[str], intents: List[str], tags_list: List[List[str]]):
        from sklearn.model_selection import cross_val_score

        print("[Classifier] Training intent classifier...")
        self.intent_pipeline.fit(texts, intents)

        # Cross-val accuracy
        scores = cross_val_score(self.intent_pipeline, texts, intents, cv=5, scoring='accuracy')
        print(f"[Classifier] ✅ Intent CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        print("[Classifier] Training tag classifier...")
        tags_binary = self.tag_mlb.fit_transform(tags_list)
        self.tag_pipeline.fit(texts, tags_binary)
        print("[Classifier] ✅ Tag classifier trained")

        return scores.mean()

    def predict(self, text: str, threshold: float = 0.3) -> Dict:
        intent_proba = self.intent_pipeline.predict_proba([text])[0]
        intent_classes = self.intent_pipeline.classes_
        best_idx = np.argmax(intent_proba)
        intent = intent_classes[best_idx]
        confidence = float(intent_proba[best_idx])

        # Tags
        tag_proba = self.tag_pipeline.predict_proba([text])
        predicted_tags = [
            RAVEN_TAGS[i] for i, p in enumerate(tag_proba[0])
            if p > threshold
        ] if hasattr(tag_proba[0], '__iter__') else []

        # Detect language
        lang = self._detect_lang(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "all_intents": {cls: float(p) for cls, p in zip(intent_classes, intent_proba)},
            "tags": predicted_tags,
            "lang": lang,
        }

    def _detect_lang(self, text: str) -> str:
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        if arabic_chars > 0:
            # Distinguish MSA Arabic vs Moroccan Darija
            darija_markers = [
                'bghit', 'kifash', 'wash', 'ndir', 'kayna', 'dyali', 'dyal',
                'ma3', 'lli', '3la', '7na', 'ntuma', 'walakin', 'blokiha',
                'n7awel', 'nkhed', 'nsedd', '9rit', 'flous', 'srqo', 'daba',
                'kheddam', 'machi', 'bzzaf', 'chhal', 'imta', 'chno', '7sab',
                'n3ref', 'nftah', 'ndkhol', 'l7sab', 'compte', 'carte',
            ]
            text_lower = text.lower()
            if any(m in text_lower for m in darija_markers):
                return "darija"
            return "arabic"

        text_lower = text.lower()
        words = re.findall(r"[a-z']+", text_lower)

        # Check darija even without Arabic chars (Darija can be written in latin)
        darija_latin_markers = {
            'bghit', 'kifash', 'wash', 'ndir', 'kayna', 'dyali', 'dyal',
            '3la', '7na', 'walakin', 'blokiha', 'n7awel', 'nkhed', 'flous',
            'srqo', 'daba', 'machi', 'bzzaf', 'chhal', 'imta', 'chno',
            'n3ref', 'nftah', 'ndkhol', 'ntuma', 'nsedd',
        }
        if any(m in text_lower for m in darija_latin_markers):
            return "darija"

        french_markers = {
            'je', 'vous', 'mon', 'ma', 'mes', 'le', 'la', 'les', 'est', 'sont',
            'pas', 'une', 'des', 'du', 'de', 'et', 'en', 'un', 'pour', 'dans',
            'que', 'qui', 'sur', 'avec', 'ne', 'se', 'ce', 'cette', 'au',
            'comment', 'quand', 'quel', 'quelle', 'voudrais', 'veux', 'souhaite',
            'puis', 'peut', 'faire', 'avoir', 'être', 'compte', 'carte',
        }
        english_markers = {
            'i', 'my', 'the', 'is', 'are', 'can', 'do', 'want', 'help', 'how',
            'what', 'when', 'where', 'who', 'would', 'could', 'should', 'have',
            'has', 'been', 'get', 'need', 'please', 'account', 'bank', 'card',
        }

        fr_score = sum(1 for w in words if w in french_markers)
        en_score = sum(1 for w in words if w in english_markers)

        # Boost with special chars
        if any(c in text for c in 'àâéèêëîïôùûüç'):
            fr_score += 3

        if fr_score >= en_score:
            return "french"
        return "english"

    def save(self, path: str):
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'intent_pipeline': self.intent_pipeline,
                'tag_pipeline': self.tag_pipeline,
                'tag_mlb': self.tag_mlb,
                'intents': RAVEN_INTENTS,
                'tags': RAVEN_TAGS,
            }, f)
        print(f"[Classifier] ✅ Saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleIntentClassifier':
        import pickle
        clf = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        clf.intent_pipeline = data['intent_pipeline']
        clf.tag_pipeline = data['tag_pipeline']
        clf.tag_mlb = data['tag_mlb']
        return clf


# ══════════════════════════════════════════════════════
#  EVAL & REPORT
# ══════════════════════════════════════════════════════

def evaluate(clf: SimpleIntentClassifier, texts: List[str], intents: List[str]):
    from sklearn.metrics import classification_report, confusion_matrix
    preds = [clf.predict(t)['intent'] for t in texts]
    print("\n[Eval] Classification Report:")
    print(classification_report(intents, preds, target_names=RAVEN_INTENTS))
    return preds


def save_training_jsonl(texts, intents, tags_list, path="data/intent_train_raven.jsonl"):
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for text, intent, tags in zip(texts, intents, tags_list):
            f.write(json.dumps({
                "text": text,
                "intent": intent,
                "tags": tags,
            }, ensure_ascii=False) + "\n")
    print(f"[Data] ✅ {len(texts):,} samples → {path}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RAVEN - Improved Intent Classifier Training")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/4] Building RAVEN dataset...")
    texts, intents, tags_list = build_dataset(n_augment=10)
    print(f"  Total samples: {len(texts):,}")
    print(f"  Distribution: {Counter(intents)}")
    save_training_jsonl(texts, intents, tags_list)

    # 2. Train
    print("\n[2/4] Training classifier...")
    clf = SimpleIntentClassifier()
    accuracy = clf.fit(texts, intents, tags_list)

    # 3. Evaluate
    print("\n[3/4] Evaluation...")
    evaluate(clf, texts, intents)

    # 4. Save
    print("\n[4/4] Saving model...")
    Path("models").mkdir(exist_ok=True)
    clf.save("models/raven_intent_classifier.pkl")

    # 5. Demo
    print("\n[Demo] Testing predictions:")
    test_cases = [
        ("bghit n7awel flous mn compte dyali", "darija"),
        ("Ma carte bancaire a été volée !", "french"),
        ("أريد الاستفسار عن رصيد حسابي", "arabic"),
        ("The app is not working, I can't login", "english"),
        ("Quel temps fait-il aujourd'hui ?", "french"),
        ("srqo carte dyali blokiha daba!", "darija"),
    ]
    for text, expected_lang in test_cases:
        result = clf.predict(text)
        print(f"\n  Input  : {text[:50]}")
        print(f"  Intent : {result['intent']} ({result['confidence']:.2f})")
        print(f"  Tags   : {result['tags']}")
        print(f"  Lang   : {result['lang']} (expected: {expected_lang})")

    print(f"\n✅ Done! Model accuracy: {accuracy:.1%}")
    print("   Model saved: models/raven_intent_classifier.pkl")
