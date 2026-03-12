
# ──────────────────────────────────────────────────────────────────────────────
# GPU CHECK
# ──────────────────────────────────────────────────────────────────────────────

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"GPU      : {torch.cuda.is_available()}")


# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import re
import pickle
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ──
DATA_PATH  = Path("C:\Users\admin\Desktop\NOUVEAU\LLM_TYBOO\step2_intent_classifier\Data\IIbalanced_shuffled_dataset.csv")  
MODEL_PATH = Path("C:\Users\admin\Desktop\NOUVEAU\LLM_TYBOO\step2_intent_classifier\model\raven_cnn_torch666.pt")

# ── Tokenizer ──
VOCAB_SIZE = 10_000
SEQ_LEN    = 96

# ── Model architecture ──
EMBED_DIM   = 64
CNN_FILTERS = 192     # per kernel → 3 × 192 = 576 features total
DROPOUT     = 0.35

# ── Training ──
EPOCHS        = 60
BATCH_SIZE    = 256
LEARNING_RATE = 2e-3
AUG_FACTOR    = 4     # augmentation multiplier on train set only
PATIENCE      = 10    # early stopping

# ── Tags ──
MIN_TAG_FREQ = 20                          # drop tags appearing < 20 times
MAX_TAG_FREQ = None                        # set after loading data

print("✅ Config ready")


# ──────────────────────────────────────────────────────────────────────────────
# TAG VOCABULARY & KEYWORD MAP
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# Complete KEYWORD_TAG_MAP (350 tags from Segmentation Excel)
# ══════════════════════════════════════════════════════════════════════════════
import openpyxl, re
from pathlib import Path
from collections import Counter

EXCEL_PATH = Path("C:\Users\admin\Desktop\NOUVEAU\LLM_TYBOO\step2_intent_classifier\Data\Segmentation_TybotSmartContact.xlsx") 

def load_tags_from_excel(path, sheet="Std1"):
    wb  = openpyxl.load_workbook(path)
    ws  = wb[sheet]
    tags = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        tag = row[3]
        if tag and str(tag).strip():
            tags.append(str(tag).strip())          # preserve original case
    tags = list(dict.fromkeys(tags))
    print(f"✅ Loaded {len(tags)} official tags from Excel")
    return tags

EXCEL_TAGS = load_tags_from_excel(EXCEL_PATH)
EXCEL_TAGS_LOWER = {t.lower(): t for t in EXCEL_TAGS}   # lookup map

# ──────────────────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION — when two opposing tags both fire, keep the stronger one
# ──────────────────────────────────────────────────────────────────────────────
CONFLICTS = {
    ("priority:high",        "priority:low"):       "priority:high",
    ("priority:high",        "priority:medium"):    "priority:high",
    ("priority:medium",      "priority:low"):       "priority:medium",
    ("urgency:high",         "urgency:low"):        "urgency:high",
    ("urgency:high",         "urgency:medium"):     "urgency:high",
    ("urgency:medium",       "urgency:low"):        "urgency:medium",
    ("budget:high",          "budget:low"):         "budget:high",
    ("budget:high",          "budget:none"):        "budget:high",
    ("budget:medium",        "budget:none"):        "budget:medium",
    ("feedback:positive",    "feedback:negative"):  "feedback:negative",
    ("loyalty:high",         "loyalty:new"):        "loyalty:high",
    ("loyalty:long-term",    "loyalty:new"):        "loyalty:long-term",
    ("need:urgent",          "need:low"):           "need:urgent",
    ("risk:high",            "risk:low"):           "risk:high",
    ("risk:high",            "risk:medium"):        "risk:high",
    ("predict:churn-high",   "predict:churn-low"):  "predict:churn-high",
}

def resolve_conflicts(tags: list) -> list:
    tag_set = set(t.lower() for t in tags)
    for (a, b), winner in CONFLICTS.items():
        if a in tag_set and b in tag_set:
            loser = b if winner == a else a
            tag_set.discard(loser)
    return list(tag_set)


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORD → TAG MAP  (covers all 350 tags from the Excel)
# Format: ([keyword_list], "tag_name")
# Keywords are matched as substrings of lowercased user_msg
# ──────────────────────────────────────────────────────────────────────────────
KEYWORD_TAG_MAP = [

    # ══════════════════════════════════════════════════════════════════════════
    # IDENTITY & PROFILE — Demographics
    # ══════════════════════════════════════════════════════════════════════════
    (["teenager", "minor", "under 18", "kid ", "enfant",
      "jeune mineur"],                                          "age: - 18"),
    (["18 years", "19 years", "20 years", "21 ans", "22 ans",
      "23 ans", "college student", "étudiant"],                "age:18-24"),
    (["25 ans", "26 ans", "28 ans", "30 ans",
      "young professional", "jeune professionnel"],            "age:25-34"),
    (["35 ans", "38 ans", "40 ans", "42 ans",
      "mid career", "milieu de carrière"],                     "age:35-44"),
    (["45 ans", "48 ans", "50 ans", "52 ans",
      "experienced professional"],                             "age:45-54"),
    (["55 ans", "58 ans", "60 ans", "62 ans",
      "pre-retirement", "proche retraite"],                    "age:55-64"),
    (["senior", "retired", "retraité", "over 65",
      "elderly", "pension", "retraite"],                       "age: + 65"),

    (["he is", "his account", "mr ", "monsieur", "sir ",
      "homme", "rajl", "رجل", "أخ "],                         "gender:male"),
    (["she is", "her account", "mrs ", "ms ", "madam",
      "madame", "femme", "مرأة", "أخت"],                      "gender:female"),
    (["non-binary", "other gender", "prefer not to say",
      "genre autre"],                                          "gender:Other"),

    (["high school", "lycée", "baccalauréat", "bac "],         "education:highschool"),
    (["bachelor", "undergraduate", "licence "],                "education:bachelor"),
    (["master", "mba", "masters degree", "master's"],          "education:master"),
    (["phd", "doctorate", "doctoral", "thèse", "thesis"],      "education:phd"),
    (["college", "community college", "associate degree"],      "education:college"),

    (["france", "paris", "lyon", "marseille",
      "french market", "marché français"],                     "country:fr"),
    (["morocco", "maroc", "المغرب", "marrocos"],               "country:ma"),
    (["spain", "españa", "espagne", "madrid", "barcelona"],    "country:es"),
    (["usa", "united states", "america", "états-unis",
      "new york", "california"],                               "country:us"),

    (["casablanca", "casa "],                                   "city:casablanca"),
    (["rabat"],                                                 "city:rabat"),
    (["paris "],                                                "city:paris"),
    (["marrakech", "marrakesh", "marrakch"],                   "city: 100 Pays 200 Villes"),

    (["bonjour", "merci beaucoup", "s'il vous plaît",
      "je veux", "je voudrais", "je suis", "mon compte",
      "ma carte", "je me", "comment puis"],                    "language:fr"),
    (["مرحبا", "شكرا", "أريد", "هل يمكن", "من فضلك",
      "أنا ", "كيف ", "ما هي", "أذكر", "أشعر"],               "language:ar"),
    (["bghit", "kifash", "wash ", "dyal", "machi",
      "bzzaf", "chhal", "l7sab", "flous", "n7awel",
      "mzyan", "waqef", "choukran"],                           "language:darija"),
    (["hola", "gracias", "quiero", "cómo",
      "español", "por favor"],                                 "language:esp"),

    # ══════════════════════════════════════════════════════════════════════════
    # IDENTITY & PROFILE — Psychographics
    # ══════════════════════════════════════════════════════════════════════════
    (["analytical", "data driven", "metrics",
      "evidence", "analyse", "analytique"],                    "personality:analytical"),
    (["creative", "design", "artistic",
      "innovation", "créatif", "imaginative"],                 "personality:creative"),
    (["social", "community", "team", "collaborate",
      "communauté", "réseau"],                                 "personality: Social"),
    (["process", "procedure", "step by step",
      "methodical", "structured"],                             "personality: Procedural"),

    (["premium", "vip", "luxury", "high end",
      "haut de gamme", "best quality"],                        "preference:premium"),
    (["affordable", "low cost", "budget",
      "économique", "pas cher", "bon marché"],                 "preference:lowcost"),
    (["innovative", "cutting edge", "latest tech",
      "modern solution"],                                       "preference: Innovative"),
    (["quality", "reliable", "trustworthy",
      "consistent", "fiable"],                                 "preference: quality"),

    (["ai ", "artificial intelligence", "automation",
      "machine learning", "chatbot", "nlp"],                   "interest: IT&AI&Automation"),
    (["technology", "tech ", "digital", "software",
      "innovation", "startup"],                                "interest: Technology&innovation"),
    (["real estate", "property", "immobilier",
      "appartement", "maison à vendre"],                       "interest:properties"),
    (["physics", "biology", "science", "research",
      "laboratory"],                                            "interest: physics & biology"),
    (["art", "design", "creative", "illustration",
      "photography", "music"],                                 "interest: Art&design"),
    (["car", "automobile", "plane", "aviation",
      "voiture", "avion"],                                     "interest: cars&plane"),

    (["performance", "results", "efficiency",
      "output", "kpi", "metrics"],                             "motivation:performance"),
    (["simple", "easy to use", "user friendly",
      "simplified", "straightforward"],                        "motivation:simplicity"),

    (["early adopter", "first to try",
      "beta tester", "new features first"],                    "attitude:early-adopter"),
    (["wait and see", "late adopter",
      "not rush to try", "proven technology"],                 "attitude:late-adopter"),

    (["expect quality", "high standard",
      "best quality", "no compromise"],                        "expectation:quality"),
    (["expect fast", "quick response",
      "fast delivery", "speed matters"],                       "expectation:speed"),

    (["premium style", "elegant", "luxury feel",
      "sophisticated"],                                         "style:premium"),
    (["basic plan", "standard", "essential",
      "no frills"],                                             "style:basic"),

    # ══════════════════════════════════════════════════════════════════════════
    # IDENTITY & PROFILE — User Role & Permissions
    # ══════════════════════════════════════════════════════════════════════════
    (["guest", "not logged in", "anonymous",
      "visiteur"],                                              "role: guest"),
    (["logged in", "authenticated", "my account",
      "mon compte", "compte dyali"],                           "role: authenticated"),
    (["identified", "verified", "know who i am"],              "role: indentified"),

    # ══════════════════════════════════════════════════════════════════════════
    # FIRMOGRAPHICS — Company Attributes
    # ══════════════════════════════════════════════════════════════════════════
    (["solo", "just me", "freelance", "1 person",
      "self employed", "seul"],                                "company-size:1-10"),
    (["small team", "small company", "small business",
      "petite équipe", "startup"],                             "company-size:10-50"),
    (["mid size", "medium company",
      "growing team", "100 employees"],                        "company-size:50-200"),
    (["large company", "500 employees",
      "corporation", "grande entreprise"],                     "company-size:200-1000"),
    (["multinational", "global company",
      "thousands of employees", "enterprise scale"],           "company-size:1000+"),

    (["revenue million", "1m revenue",
      "small revenue"],                                         "revenue:1m"),
    (["5 million", "5m revenue",
      "mid revenue"],                                           "revenue:5m"),
    (["20 million", "20m+", "large revenue",
      "high revenue"],                                          "revenue:20m+"),

    (["local market", "marché local",
      "regional business", "local only"],                      "market:local"),
    (["global market", "international",
      "worldwide", "multinational"],                           "market:global"),
    (["regional", "north africa", "mena",
      "regional expansion"],                                   "market:regional"),

    (["fast growing", "rapid growth",
      "scale fast", "high growth"],                            "growth:fast"),
    (["stable", "steady", "consistent growth",
      "mature company"],                                        "growth:stable"),

    (["private company", "privately held",
      "not listed", "société privée"],                         "ownership:private"),
    (["public company", "listed", "stock exchange",
      "société cotée"],                                         "ownership:public"),

    (["b2b", "business to business",
      "enterprise client", "sell to businesses"],              "model:b2b"),
    (["b2c", "consumer", "end user",
      "individual customer", "retail customer"],               "model:b2c"),
    (["government", "public sector",
      "ministry", "administration"],                           "model: b2g"),
    (["service company", "service based",
      "consulting firm"],                                       "model:service"),
    (["integration", "api business",
      "platform integration"],                                  "model: integration"),
    (["distribution", "reseller",
      "distributor", "distributeur"],                          "model: distribution"),

    # ══════════════════════════════════════════════════════════════════════════
    # FIRMOGRAPHICS — Industry
    # ══════════════════════════════════════════════════════════════════════════
    (["bank", "banking", "finance", "financial",
      "investment", "fintech", "trading", "bourse",
      "banque", "lbank"],                                       "industry:finance"),
    (["health", "hospital", "medical", "clinic",
      "healthcare", "pharma", "doctor"],                        "industry:health"),
    (["retail", "store", "ecommerce", "shop",
      "boutique", "magasin"],                                   "industry:retail"),
    (["manufacturing", "factory", "production",
      "industrial", "usine"],                                   "industry:manufacturing"),
    (["service", "consulting", "advisory",
      "professional services"],                                 "industry: Service"),
    (["agriculture", "farming", "agri",
      "crops", "livestock"],                                    "industry:agriculture"),
    (["telecom", "telco", "mobile network",
      "internet provider", "isp"],                             "industry:telco"),
    (["insurance", "assurance", "policy",
      "coverage", "claim"],                                     "industry:insurance"),
    (["school", "university", "education sector",
      "edtech", "academia", "training"],                       "industry:education"),
    (["transport", "logistics", "shipping",
      "delivery", "fleet"],                                     "industry:transport"),
    (["energy", "electricity", "solar",
      "oil", "gas", "renewable"],                              "industry:energy"),
    (["gdpr", "data protection",
      "privacy regulation", "compliance"],                     "regulation:gdpr"),
    (["hipaa", "medical regulation",
      "health compliance"],                                     "regulation:hipaa"),
    (["public sector", "état", "gouvernement",
      "public institution"],                                   "sector:public"),
    (["private sector", "secteur privé",
      "private enterprise"],                                   "sector:private"),

    # ══════════════════════════════════════════════════════════════════════════
    # FIRMOGRAPHICS — Job Role & Seniority
    # ══════════════════════════════════════════════════════════════════════════
    (["ceo", "chief executive", "founder",
      "co-founder", "managing director"],                      "role:ceo"),
    (["cto", "chief technology", "tech lead",
      "vp engineering"],                                        "role:cto"),
    (["cfo", "chief financial", "finance director",
      "vp finance"],                                            "role:cfo"),
    (["cmo", "chief marketing", "vp marketing",
      "marketing director"],                                    "role:cmo"),
    (["product manager", "product owner",
      "pm ", "chef de produit"],                               "role:product-manager"),
    (["developer", "programmer", "backend",
      "frontend", "fullstack", "software engineer"],           "role:developer"),
    (["data scientist", "data analyst",
      "bi analyst", "data engineer"],                          "role:data-analyst"),
    (["designer", "ux ", "ui designer",
      "graphic designer", "créatif"],                          "role:designer"),
    (["consultant", "advisor", "conseiller"],                  "role:consultant"),
    (["engineer", "ingénieur", "technical"],                   "role:engineer"),
    (["administrator", "admin ", "sys admin",
      "it admin"],                                              "role:administrator"),

    (["executive", "c-level", "c level",
      "senior leadership"],                                     "seniority:executive"),
    (["director", "directeur", "vp "],                         "seniority:director"),
    (["manager", "chef d'équipe", "team lead"],                "seniority:manager"),
    (["staff", "employee", "team member",
      "individual contributor"],                               "seniority:staff"),

    (["it department", "it team",
      "tech department"],                                       "department:it"),
    (["finance department", "accounting",
      "comptabilité"],                                          "department:finance"),
    (["marketing team", "marketing department",
      "équipe marketing"],                                      "department:marketing"),

    # ══════════════════════════════════════════════════════════════════════════
    # FIRMOGRAPHICS — Buying Center (BANT)
    # ══════════════════════════════════════════════════════════════════════════
    (["i decide", "i will buy", "decision maker",
      "i approve", "je décide"],                               "buyer:buyer"),
    (["i recommend", "i suggested",
      "i referred", "j'ai recommandé"],                        "buyer: recommender"),
    (["i initiated", "i brought this up",
      "i started the process"],                                 "buyer: initiator"),
    (["i influence", "stakeholder",
      "i have a say"],                                          "buyer: influencer"),
    (["final decision", "i approved",
      "i authorized"],                                          "buyer: decider"),
    (["end user", "i use it daily",
      "i'm the one using it"],                                  "buyer role: End user"),

    (["no budget", "can't afford",
      "out of budget", "pas de budget"],                       "budget:none"),
    (["small budget", "limited budget",
      "tight budget", "budget serré"],                         "budget:low"),
    (["reasonable budget", "moderate budget",
      "mid budget"],                                            "budget:medium"),
    (["large budget", "invest heavily",
      "high budget", "significant investment"],                 "budget:high"),

    (["i decide alone", "full authority",
      "i have authority"],                                      "authority:high"),
    (["some authority", "partial decision",
      "share decision"],                                        "authority:medium"),
    (["limited authority", "need approval"],                   "authority:low"),
    (["no authority", "just an end user",
      "not the decision maker"],                               "authority:none"),

    (["urgent need", "need it now",
      "critical requirement", "besoin urgent"],                "need:urgent"),
    (["need it soon", "medium priority need"],                  "need:medium"),
    (["low priority", "not critical",
      "nice to have", "someday"],                              "need:low"),
    (["no need", "not looking for",
      "don't need"],                                            "need:none"),

    (["need it now", "immediately",
      "this week", "timing:now"],                              "timing:now"),
    (["in 3 months", "next quarter",
      "within 3 months"],                                      "timing:3months"),
    (["in 6 months", "second half",
      "within 6 months"],                                      "timing:6months"),

    # ══════════════════════════════════════════════════════════════════════════
    # NEEDS & PREDICTIVE — Intent Signals
    # ══════════════════════════════════════════════════════════════════════════
    (["want to buy", "ready to purchase",
      "looking to get", "i want to order",
      "i want to subscribe"],                                   "intent:buy"),
    (["researching", "gathering info",
      "learning about", "trying to understand",
      "just looking"],                                          "intent:research"),
    (["comparing", "which is better",
      "alternatives", "vs ", "compare options",
      "difference between"],                                   "intent:compare"),
    (["show me a demo", "can i see it",
      "demonstration", "demo please"],                         "intent:demo"),
    (["free trial", "trial period",
      "try before i buy", "essai gratuit"],                    "intent:trial"),
    (["upgrade my plan", "upgrade account",
      "move to higher tier"],                                  "intent:upgrade"),
    (["renew", "renewal", "extend my plan",
      "renouveler"],                                            "intent:renew"),
    (["contact you", "reach out",
      "get in touch", "contacter"],                            "intent:contact"),
    (["pricing", "how much", "cost",
      "tarif", "prix", "combien ça coûte",
      "chhal", "thaman"],                                       "intent:pricing"),
    (["integrate", "integration",
      "connect with", "sync with", "api"],                     "intent:integration"),

    # ══════════════════════════════════════════════════════════════════════════
    # NEEDS & PREDICTIVE — Pain Points
    # ══════════════════════════════════════════════════════════════════════════
    (["manual process", "doing it manually",
      "by hand", "takes too long"],                            "pain:manual-process"),
    (["slow", "too slow", "lag", "bottleneck",
      "inefficient", "takes forever"],                         "pain:slow-ops"),
    (["data silo", "isolated data",
      "can't share data", "disconnected systems"],             "pain:data-silos"),
    (["pipeline issue", "broken pipeline",
      "pipeline problem"],                                      "pain:pipeline"),
    (["automate", "automation",
      "automatic", "workflow"],                                 "need:automation"),
    (["ai ", "artificial intelligence",
      "smart system", "machine learning"],                     "need:ai"),
    (["integrate", "integration", "connect",
      "sync", "api ", "webhook"],                              "need:integration"),
    (["insights", "analytics", "reporting",
      "dashboard", "data visibility"],                         "need:insights"),
    (["training", "onboarding support",
      "learn how to use", "formation"],                        "need:training"),
    (["governance", "compliance framework",
      "policy enforcement"],                                    "need:governance"),

    (["urgent", "asap", "immediately",
      "right now", "critical", "today",
      "3ajil", "daba", "fawran", "فوراً"],                    "urgency:high"),
    (["soon", "this week", "in a few days",
      "by friday", "shortly"],                                 "urgency:medium"),
    (["no rush", "whenever", "not urgent",
      "flexible", "whenever you can"],                         "urgency:low"),

    (["top priority", "most important",
      "critical", "must have", "highest priority"],            "priority:high"),
    (["medium priority", "important but not critical"],         "priority:medium"),
    (["low priority", "nice to have",
      "not critical", "minor issue"],                          "priority:low"),

    (["error", "bug", "broken",
      "not working", "crash", "glitch",
      "doesn't work", "machi khdama"],                        "pain:errors"),
    (["too expensive", "overcost",
      "high cost", "costly", "ghali bzzaf"],                   "pain:overcost"),
    (["complex", "complicated",
      "confusing", "hard to use", "difficult"],                "pain:complexity"),
    (["migrate", "migration",
      "switch from", "move from"],                             "need:migration"),
    (["security", "secure", "privacy",
      "protect data", "encryption",
      "block card", "bloquer", "waqaf"],                       "need:security"),
    (["performance", "speed up",
      "faster system", "optimize"],                             "need:performance"),
    (["scale", "scalability",
      "grow", "handle more users"],                            "need:scalability"),

    # ══════════════════════════════════════════════════════════════════════════
    # NEEDS & PREDICTIVE — Fit & Readiness
    # ══════════════════════════════════════════════════════════════════════════
    (["perfect fit", "exactly what i need",
      "ideal solution"],                                        "fit:strong"),
    (["not a great fit", "not ideal",
      "partially fits"],                                        "fit:weak"),
    (["good enough", "reasonable fit",
      "mostly fits"],                                           "fit:medium"),

    (["ready to start", "let's go",
      "ready now"],                                             "readiness:now"),
    (["not ready yet", "need more time",
      "later", "pas encore"],                                  "readiness:later"),

    # ══════════════════════════════════════════════════════════════════════════
    # NEEDS & PREDICTIVE — Predictive Scores
    # ══════════════════════════════════════════════════════════════════════════
    (["cancel my account", "closing my account",
      "leaving you", "unsubscribe",
      "will post on social media",
      "going to complain publicly",
      "social media threat"],                                   "predict:churn-high"),
    (["thinking of leaving", "not happy",
      "considering cancelling",
      "disappointed with service"],                            "predict:churn-medium"),
    (["happy customer", "staying",
      "satisfied", "renewing"],                                "predict:churn-low"),

    (["want to upgrade", "add more features",
      "expand plan", "need more"],                             "predict:upsell-high"),
    (["current plan is fine",
      "don't need more"],                                      "predict:upsell-low"),

    (["ready to buy", "going to purchase",
      "decided to get"],                                        "predict:conversion-high"),
    (["just browsing", "not sure yet",
      "still thinking"],                                        "predict:conversion-low"),

    (["loyal customer", "long time",
      "years with you", "always come back"],                   "score:loyalty"),
    (["engaged", "active user",
      "using it regularly"],                                   "score:engagement"),
    (["healthy", "all good",
      "no issues", "working fine"],                            "score:health"),

    (["very risky", "high risk",
      "critical issue"],                                        "risk:high"),
    (["some risk", "moderate risk",
      "medium risk"],                                           "risk:medium"),
    (["low risk", "minimal risk",
      "stable situation"],                                      "risk:low"),

    (["great opportunity", "high potential",
      "big deal"],                                              "opportunity:high"),
    (["some opportunity", "moderate potential"],               "opportunity:medium"),
    (["small opportunity", "low potential"],                   "opportunity:low"),

    (["positive outlook", "going well",
      "optimistic", "confident"],                              "forecast:positive"),
    (["pessimistic", "declining",
      "worried about outcome", "not looking good"],            "forecast:negative"),

    (["about to buy", "ready to purchase",
      "going to order"],                                        "propensity:buy"),
    (["thinking about upgrading",
      "might upgrade"],                                         "propensity:upgrade"),

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIORAL & ENGAGEMENT — Usage Intensity
    # ══════════════════════════════════════════════════════════════════════════
    (["every day", "daily", "constantly",
      "all the time", "chaque jour"],                          "sessions:daily"),
    (["every week", "weekly",
      "few times a week", "chaque semaine"],                   "sessions:weekly"),
    (["monthly", "once a month",
      "occasionally", "chaque mois"],                          "sessions:monthly"),

    (["power user", "heavy user",
      "use it all the time", "rely on it daily"],              "usage:high"),
    (["moderate use", "use it regularly"],                     "usage:medium"),
    (["rarely use", "sometimes", "barely use",
      "not often"],                                             "usage:low"),

    (["active last 7 days", "used it today",
      "just logged in"],                                        "active:7days"),
    (["active last month", "used it this month"],              "active:30days"),

    (["power user", "advanced user",
      "use everything", "all features"],                       "engagement:power-user"),
    (["occasional user", "use it sometimes",
      "not very active"],                                       "engagement:occasional"),

    (["login frequently", "log in every day"],                 "login:frequent"),
    (["rarely login", "haven't logged in",
      "forgot my login"],                                       "login:rare"),

    (["heavy consumption", "use a lot",
      "high volume"],                                           "consumption:high"),
    (["low consumption", "light use",
      "minimal usage"],                                         "consumption:low"),

    (["very sticky", "can't live without",
      "use it every day"],                                      "stickiness:high"),
    (["not sticky", "easy to replace",
      "don't depend on it"],                                   "stickiness:low"),

    (["deep user", "use advanced features",
      "expert user"],                                           "depth:deep"),
    (["basic user", "surface level",
      "only basic features"],                                  "depth:shallow"),

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIORAL & ENGAGEMENT — Engagement Level
    # ══════════════════════════════════════════════════════════════════════════
    (["opened your email", "read your email",
      "clicked the email"],                                     "email:opened"),
    (["clicked the link", "clicked your email"],               "email:clicked"),
    (["ignored your email", "never open emails",
      "spam"],                                                  "email:ignored"),

    (["push notification", "allow notifications",
      "notifications enabled"],                                 "push:enabled"),
    (["disabled notifications", "turned off push",
      "no notifications"],                                      "push:disabled"),

    (["active on web", "using the website",
      "on your website"],                                       "web:active"),
    (["not using website", "rarely visit"],                    "web:inactive"),

    (["loyal", "long time customer",
      "always come back", "since day one",
      "7 snin m3akom", "années avec vous"],                    "loyalty:high"),
    (["somewhat loyal", "usually come back"],                  "loyalty:medium"),
    (["not loyal", "first time", "just started"],              "loyalty:low"),
    (["long term customer", "years with you",
      "fidèle depuis longtemps"],                              "loyalty:long-term"),
    (["new customer", "just joined",
      "recently signed up", "first time"],                     "loyalty:new"),

    (["responded to promo", "used discount",
      "reacted to offer"],                                      "promotion:responsive"),
    (["ignored promo", "didn't use discount"],                 "promotion:unresponsive"),

    (["attended event", "at the event",
      "was there"],                                             "event:attendee"),
    (["missed event", "couldn't attend"],                      "event:absent"),

    (["active on social", "social media post",
      "shared on social"],                                      "social:active"),
    (["not on social", "no social media"],                     "social:inactive"),

    (["contacted support", "support ticket",
      "opened a ticket", "support request"],                   "support:active"),
    (["never contact support", "no support needed"],           "support:inactive"),

    (["love it", "amazing", "excellent",
      "great product", "very happy", "satisfied",
      "works perfectly", "highly recommend",
      "mzyan bzzaf", "3jbatni", "رائع", "أحسنتم"],            "feedback:positive"),
    (["disappointed", "frustrated", "terrible",
      "worst", "unacceptable", "angry", "hate it",
      "never works", "awful", "horrible",
      "mherres", "متضايق", "غاضب",
      "laisse à désirer", "abandoned"],                        "feedback:negative"),

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIORAL & ENGAGEMENT — Lifecycle Stage
    # ══════════════════════════════════════════════════════════════════════════
    (["potential customer", "interested lead",
      "prospect lead"],                                         "stage:lead"),
    (["marketing qualified", "mql",
      "responded to campaign"],                                 "stage:mql (marketing qualified)"),
    (["sales qualified", "sql",
      "ready for sales"],                                       "stage:sql (sales qualified)"),
    (["prospect", "evaluating",
      "considering your service"],                              "stage:prospect"),
    (["customer", "client", "account holder",
      "user of your service"],                                  "stage:customer"),
    (["active customer", "currently using",
      "daily active"],                                          "stage:active"),
    (["inactive", "stopped using",
      "haven't used in a while"],                              "stage:inactive"),
    (["cancel", "leaving", "might cancel",
      "social media threat", "if you don't fix",
      "ila ma7lawtich", "إذا لم تحلوا",
      "taking my business elsewhere"],                         "stage:churn-risk"),
    (["churned", "left your service",
      "cancelled account"],                                     "stage:churned"),
    (["advocate", "ambassador",
      "recommend to everyone", "refer friends"],               "stage:advocate"),
    (["expansion", "want more",
      "add more users"],                                        "stage:expansion"),
    (["just signed up", "new here",
      "getting started", "onboarding",
      "first time using"],                                      "stage:onboarding"),
    (["nurturing", "still deciding",
      "need more info before deciding"],                       "stage:nurturing"),
    (["upgrade", "add more", "expand my plan",
      "upsell"],                                                "stage:upsell"),
    (["renewal", "renew my plan",
      "extend subscription"],                                  "stage:renewal"),
    (["lost deal", "went with competitor",
      "chose another"],                                         "stage:lost"),
    (["won deal", "signed contract",
      "deal closed"],                                           "stage:win"),
    (["opportunity", "potential deal",
      "sales opportunity"],                                     "stage:opportunity"),
    (["on trial", "free trial",
      "testing your product"],                                  "stage:trial"),
    (["evaluating", "in evaluation phase",
      "still testing"],                                         "stage:evaluation"),

    # ══════════════════════════════════════════════════════════════════════════
    # BEHAVIORAL & ENGAGEMENT — Channel Interaction
    # ══════════════════════════════════════════════════════════════════════════
    (["via email", "by email", "send email",
      "email me", "par email"],                                 "channel:email"),
    (["sms", "text message", "texto"],                         "channel:sms"),
    (["push notification", "app notification"],                "channel:push"),
    (["whatsapp", "send me on whatsapp",
      "text me on whatsapp"],                                  "channel:whatsapp"),
    (["web chat", "chat on website",
      "live chat"],                                             "channel:webChat"),
    (["instagram", "dm on instagram",
      "instagram message"],                                    "channel: Instagram"),
    (["facebook", "messenger", "fb message",
      "facebook message"],                                     "channel: Facebook"),
    (["telegram", "telegram message"],                         "channel: Telegram"),

    (["on mobile", "phone app",
      "mobile app", "using my phone"],                         "device:mobile"),
    (["on desktop", "laptop", "computer",
      "browser", "pc "],                                        "device:desktop"),

    (["ios", "iphone", "apple device"],                        "os:ios"),
    (["android", "samsung", "huawei"],                         "os:android"),

    (["direct visit", "typed the url",
      "came directly"],                                         "traffic:direct"),
    (["google search", "organic search",
      "found you online"],                                      "traffic:organic"),
    (["paid ad", "saw your ad",
      "clicked an ad"],                                         "traffic:paid"),
    (["referral", "someone referred",
      "friend told me"],                                        "traffic:referral"),

    (["from campaign", "email campaign",
      "marketing campaign"],                                   "source:campaign"),
    (["from ads", "saw your advertisement"],                   "source:ads"),
    (["outbound call", "you contacted me",
      "cold call"],                                             "source: outbound"),
    (["influencer", "saw influencer",
      "recommended by creator"],                               "source: influencer"),
    (["newsletter", "your newsletter",
      "subscribed to newsletter"],                             "source:newsletter"),

    # ══════════════════════════════════════════════════════════════════════════
    # TRANSACTIONAL & REVENUE — RFM
    # ══════════════════════════════════════════════════════════════════════════
    (["bought last week", "purchased recently",
      "last 7 days"],                                           "recency:7d"),
    (["bought last month", "purchased this month",
      "last 30 days"],                                          "recency:30d"),
    (["bought 3 months ago",
      "last 90 days"],                                          "recency:90d"),
    (["bought 6 months ago",
      "last 180 days"],                                         "recency:180d"),

    (["buy occasionally", "1 to 5 purchases"],                 "frequency:1-5"),
    (["buy regularly", "5 to 10 purchases"],                   "frequency:5-10"),
    (["buy very often", "10+ purchases",
      "frequent buyer"],                                        "frequency:10+"),

    (["low spender", "small purchases",
      "low value customer"],                                    "monetary:low"),
    (["medium spender", "average value"],                      "monetary:medium"),
    (["high spender", "big purchases",
      "high value customer"],                                   "monetary:high"),

    (["vip customer", "vip client",
      "top customer"],                                          "rfm:vip"),
    (["at risk customer", "losing them",
      "at risk of churn"],                                      "rfm:at-risk"),
    (["new customer", "first purchase",
      "just joined"],                                           "rfm:new"),
    (["champion customer", "best customer",
      "top tier client"],                                       "rfm:champion"),
    (["inactive customer", "stopped buying",
      "dormant"],                                               "rfm:inactive"),

    (["short sales cycle", "quick purchase",
      "fast decision"],                                         "cycle:short"),
    (["medium sales cycle", "few weeks to decide"],            "cycle:medium"),
    (["long sales cycle", "takes months to decide",
      "complex procurement"],                                   "cycle:long"),

    (["high value", "premium client",
      "big account"],                                           "value:high"),
    (["low value", "small account",
      "minor client"],                                          "value:low"),
    (["medium value", "average account"],                      "value:Medium"),

    # ══════════════════════════════════════════════════════════════════════════
    # TRANSACTIONAL & REVENUE — Purchase Behavior
    # ══════════════════════════════════════════════════════════════════════════
    (["one time purchase", "buy once",
      "single order"],                                          "purchase:one-time"),
    (["repeat purchase", "bought again",
      "second time buying"],                                   "purchase:repeat"),
    (["subscription", "monthly plan",
      "recurring payment", "annual plan",
      "abonnement"],                                            "purchase:subscription"),
    (["seasonal", "holiday purchase",
      "summer only", "seasonal buyer"],                        "purchase:seasonal"),
    (["impulse buy", "spontaneous purchase",
      "bought on a whim"],                                      "purchase:impulse"),
    (["planned purchase", "researched before buying",
      "deliberate choice"],                                     "purchase:planned"),

    (["cart abandoned", "didn't complete",
      "left checkout"],                                         "cart:abandoned"),
    (["completed checkout", "purchase completed",
      "order placed"],                                          "cart:completed"),

    (["too expensive", "price too high",
      "can't afford", "reduce fees", "lower the price",
      "discount", "khfiw liya", "frais élevés",
      "rsom ghalia"],                                           "price-sensitivity:high"),
    (["price is fine", "affordable",
      "good value", "worth it",
      "prix raisonnable"],                                      "price-sensitivity:low"),

    (["core product", "main product",
      "primary plan"],                                          "product:core"),
    (["addon", "add-on", "extra feature",
      "additional module"],                                     "product:addons"),

    (["used a discount", "coupon code",
      "promo code"],                                            "discount:user"),
    (["no discount", "paid full price",
      "no coupon"],                                             "discount:non-user"),

    (["refund", "money back", "return my money",
      "chargeback", "remboursement",
      "استرداد", "rad lflous"],                                "refund:yes"),

    (["returning customer", "came back",
      "returning buyer"],                                       "returning:yes"),
    (["first time buyer", "new buyer"],                        "returning:no"),

    (["large basket", "big order",
      "high cart value"],                                       "basket:high"),
    (["small basket", "small order",
      "low cart value"],                                        "basket:low"),

    # ══════════════════════════════════════════════════════════════════════════
    # TRANSACTIONAL & REVENUE — Customer Value Tier
    # ══════════════════════════════════════════════════════════════════════════
    (["bronze", "entry level",
      "basic tier"],                                            "tier:bronze"),
    (["silver", "mid tier",
      "standard tier"],                                         "tier:silver"),
    (["gold", "premium tier",
      "gold member"],                                           "tier:gold"),
    (["platinum", "top tier",
      "platinum member", "vip"],                               "tier:platinum"),
    (["enterprise", "large account",
      "enterprise plan"],                                       "tier:enterprise"),
    (["startup plan", "startup tier",
      "early stage company"],                                   "tier:startup"),

    (["profitable customer", "high margin",
      "good roi"],                                              "profitability:high"),
    (["low profit", "low margin",
      "not profitable"],                                        "profitability:low"),

    (["positive margin", "making money",
      "margin positive"],                                       "margin:positive"),
    (["negative margin", "losing money",
      "margin negative"],                                       "margin:negative"),

    (["short relationship", "recent customer",
      "joined recently"],                                       "customer-lifetime:short"),
    (["long relationship", "years with us",
      "long time customer"],                                    "customer-lifetime:long"),

    (["likely to renew", "will renew",
      "planning to renew"],                                     "renewal-likelihood:high"),
    (["might not renew", "uncertain renewal",
      "considering cancellation"],                              "renewal-likelihood:low"),

    (["growing account", "expanding usage",
      "high growth customer"],                                  "growth:high"),
    (["shrinking account", "less usage",
      "declining customer"],                                    "growth:low"),

    (["premium value", "premium client",
      "top value customer"],                                    "value:premium"),
    (["standard value", "normal customer",
      "average account"],                                       "value:standard"),

    # ══════════════════════════════════════════════════════════════════════════
    # BANKING SPECIFIC (domain-level signals, mapped to closest Excel tags)
    # ══════════════════════════════════════════════════════════════════════════
    (["block my card", "bloquer ma carte",
      "card blocked", "carte bloquée",
      "waqaf lbital", "bloquer carte",
      "إيقاف بطاقتي"],                                         "need:security"),
    (["account statement", "relevé de compte",
      "kchf l7sab", "كشف حساب",
      "bank statement", "extrait de compte"],                  "intent:contact"),
    (["transfer money", "virement",
      "n7awel flous", "تحويل",
      "send money", "l7awala"],                                "intent:contact"),
    (["rib", "account number",
      "numéro de compte", "rqm l7sab",
      "iban"],                                                  "intent:contact"),
    (["reset password", "forgot password",
      "mot de passe oublié", "pin oublié",
      "oublié mon pin", "réinitialiser"],                      "need:security"),
    (["loan", "crédit", "qard",
      "قرض", "mortgage",
      "crédit immobilier", "financement"],                     "intent:pricing"),
    (["savings account", "compte épargne",
      "tawfir", "توفير", "saving plan"],                       "intent:research"),
    (["appointment", "rendez-vous",
      "maw3id", "موعد"],                                       "intent:contact"),
    (["activate card", "carte activée",
      "activation", "تفعيل"],                                  "intent:contact"),
    (["not resolved", "toujours pas réglé",
      "ma t7alach", "لم تُحل",
      "third time", "marra tanya",
      "المرة الثالثة"],                                         "urgency:high"),
    (["social media threat", "f facebook",
      "réseaux sociaux", "وسائل التواصل",
      "post about you", "nktb 3liha"],                         "stage:churn-risk"),
    (["recommend to friends", "n9ul ls7abi",
      "أوصي بكم", "refer friends",
      "tell everyone about you"],                               "stage:advocate"),
    (["follow up", "any update",
      "what happened to my",
      "ach wqe3 f", "ما الجديد",
      "any news on my", "status of my request"],               "stage:active"),
    (["complaint", "plainte", "chikaya",
      "شكوى", "ticket", "reported issue"],                     "feedback:negative"),

    # ── Tags that can be inferred from text (previously missing) ──
    (["english", "in english", "speak english",
      "answer in english"],                                     "language:en"),
    (["agriculture", "farming sector",
      "primary sector", "agri sector"],                        "industry:primary sectors"),
    (["not interested", "no need",
      "don't want", "not buying",
      "je ne suis pas intéressé"],                             "buyer:no"),
    (["no timing", "no deadline",
      "whenever", "no rush at all"],                           "timing : none"),
    (["not a fit", "doesn't fit",
      "not suitable"],                                          "fit: none"),
    (["submitted a refund claim",
      "no refund", "refund denied",
      "refund rejected"],                                       "refund:no"),
    (["recurring payment", "pay every month",
      "monthly payment", "paiement récurrent"],                "purchase:reccurent"),
    (["filled my cart", "added to cart",
      "cart is full"],                                          "cart: fill"),
    (["submitted via form", "filled a form",
      "via landing page", "from landing page"],                "channel: LandingPage"),
    (["other channel", "via other means",
      "different channel"],                                     "channel: others"),
    (["use 10 features", "events 10",
      "10 actions"],                                            "events:10+"),
    (["use 100 features", "events 100",
      "100 actions"],                                           "events:100+"),
    (["basic features only", "starter features",
      "few features"],                                          "feature-depth:basic"),
    (["medium features", "several features",
      "moderate adoption"],                                     "feature-depth:medium"),
]


# ──────────────────────────────────────────────────────────────────────────────
# ENRICHMENT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def enrich_tags_from_text(user_msg, existing_tag_str):
    """
    Priority:
    1. CSV tag exists and is a valid Excel tag → keep it
    2. CSV tag invalid or empty → keyword matching
    3. No keyword match → 'untagged'
    Language tags are determined by script detection (Arabic chars)
    to avoid false positives from substring matching.
    """
    existing_raw  = str(existing_tag_str).strip().lower()
    existing_tags = [t.strip() for t in existing_raw.split(",")
                     if t.strip() not in ("", "nan", "none", "null", "untagged")]
    valid_from_csv = [t for t in existing_tags
                      if t in EXCEL_TAGS_LOWER]

    if valid_from_csv:
        return ", ".join(valid_from_csv)

    text_lower = str(user_msg).lower()

    # ── Script-based language detection (overrides keyword rules) ──
    arabic_chars = len(re.findall(r"[\u0600-\u06ff]", str(user_msg)))
    darija_words  = ["bghit", "dyal", "wash", "bzzaf", "mzyan",
                     "n7awel", "flous", "l7sab", "choukran", "waqef"]
    is_darija     = any(w in text_lower for w in darija_words)
    """"""
        # Guard — if message has Arabic script, never assign language:fr or language:en
    # from keyword matching (script is more reliable than substring keywords)
    arabic_chars = len(re.findall(r"[\u0600-\u06ff]", str(user_msg)))
    if arabic_chars > 3:
        # Force skip all language: keyword rules — handled by script detection
        skip_prefixes = {"language:fr", "language:en", "language:esp"}
    else:
        skip_prefixes = set()


    script_tags   = []
    if is_darija:
        script_tags.append("language:darija")
    elif arabic_chars > 3:
        script_tags.append("language:ar")

    matched = []
    for keywords, tag in KEYWORD_TAG_MAP:
        tag_lower = tag.lower()
        # Skip language tags — handled by script detection
        if tag_lower.startswith("language:") or tag_lower in skip_prefixes:
            continue
        if any(kw in text_lower for kw in keywords):
            # Only assign if the tag exists in Excel (case-insensitive)
            canonical = EXCEL_TAGS_LOWER.get(tag_lower)
            if canonical:
                matched.append(canonical)

    matched = script_tags + matched
    matched = resolve_conflicts(matched)

    # Deduplicate preserving order
    seen, deduped = set(), []
    for t in matched:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return ", ".join(deduped) if deduped else "untagged"


print(f"✅ Tag enricher ready")
print(f"   Official Excel tags : {len(EXCEL_TAGS)}")
print(f"   Keyword rules       : {len(KEYWORD_TAG_MAP)}")
print(f"   Conflict rules      : {len(CONFLICTS)}")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD & CLEAN CSV
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

print(f"Raw rows : {len(df):,}")
print(f"Columns  : {list(df.columns)}")

# ── Clean ──
df.dropna(subset=["user_msg", "intent"], inplace=True)
df["user_msg"] = df["user_msg"].astype(str).str.strip()
df["bot_msg"]  = df.get("bot_msg", pd.Series([""] * len(df))).fillna("").astype(str)
df["intent"]   = (df["intent"]
                  .astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_"))
df["tag"]      = df["tag"].fillna("").astype(str).str.strip().str.lower()

# ── Enrich tags (CSV tag wins, keyword map is fallback) ──
df["tag"] = df.apply(
    lambda row: enrich_tags_from_text(row["user_msg"], row["tag"]),
    axis=1
)

# Drop rows with empty messages
df = df[df["user_msg"].str.len() > 3].reset_index(drop=True)

print(f"Rows after cleaning : {len(df):,}")
print(f"\nIntent distribution:")
for intent, count in df["intent"].value_counts().items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {intent:<30} {count:>6,}  ({pct:.1f}%)  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# BUILD TAG VOCABULARY
# ──────────────────────────────────────────────────────────────────────────────

def parse_tags(tag_str):
    return [t.strip() for t in str(tag_str).split(",")
            if t.strip() and t.strip() not in ("nan", "none", "null", "untagged")]


# Count all tags
all_tags_counter = Counter()
for tag_str in df["tag"]:
    all_tags_counter.update(parse_tags(tag_str))

total_rows   = len(df)
MIN_TAG_FREQ = 20
MAX_TAG_FREQ = int(total_rows * 0.40)   # drop tags appearing in > 40% of rows (noise)

TAG_VOCAB = sorted(
    t for t, c in all_tags_counter.items()
    if MIN_TAG_FREQ <= c <= MAX_TAG_FREQ
)
N_TAGS    = len(TAG_VOCAB)
TAG_INDEX = {t: i for i, t in enumerate(TAG_VOCAB)}

print(f"Total unique tags      : {len(all_tags_counter)}")
print(f"Tags kept ({MIN_TAG_FREQ}–{MAX_TAG_FREQ} freq): {N_TAGS}")
print(f"Tags dropped too rare  : "
      f"{sum(1 for t,c in all_tags_counter.items() if c < MIN_TAG_FREQ)}")
print(f"Tags dropped too common: "
      f"{sum(1 for t,c in all_tags_counter.items() if c > MAX_TAG_FREQ)}")

print(f"\nFinal tag vocab:")
for tag in TAG_VOCAB:
    cnt = all_tags_counter[tag]
    print(f"  {tag:<40} {cnt:>6,}  ({cnt/total_rows:.1%})")


def encode_tags(tag_str):
    tags = parse_tags(tag_str)
    vec  = np.zeros(N_TAGS, dtype=np.float32)
    for t in tags:
        if t in TAG_INDEX:
            vec[TAG_INDEX[t]] = 1.0
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# INTENT ENCODER
# ──────────────────────────────────────────────────────────────────────────────

INTENTS       = sorted(df["intent"].unique().tolist())
N_INTENTS     = len(INTENTS)
intent_encoder = LabelEncoder()
intent_encoder.fit(INTENTS)

print(f"✅ {N_INTENTS} intents:")
for i, name in enumerate(intent_encoder.classes_):
    count = (df["intent"] == name).sum()
    print(f"  {i:>2}  {name:<30} {count:>6,} rows")


# ──────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def augment_text(text):
    words    = str(text).split()
    suffixes = ["", "?", "!", ".", " please", " merci",
                " شكراً", " baraka", " stp", " svp"]
    r = random.random()

    if r < 0.15 and len(words) > 4:
        w = words[:]
        w.pop(random.randint(0, len(w) - 1))
        return " ".join(w)
    elif r < 0.28:
        return text.rstrip("?!.،") + random.choice(suffixes)
    elif r < 0.40:
        return text.lower()
    elif r < 0.52 and len(words) > 2:
        w = words[:]
        i = random.randint(0, len(w) - 2)
        w[i], w[i + 1] = w[i + 1], w[i]
        return " ".join(w)
    elif r < 0.64:
        return text + " " + random.choice(
            ["", "merci", "please", "شكراً", "baraka", "stp"])
    elif r < 0.78 and len(words) > 3:
        keep = max(2, int(len(words) * random.uniform(0.6, 0.9)))
        return " ".join(words[:keep])
    else:
        return text


def augment_dataframe(df, factor=AUG_FACTOR):
    rows = []
    for _, row in df.iterrows():
        for _ in range(factor):
            rows.append({
                "user_msg": augment_text(row["user_msg"]),
                "intent":   row["intent"],
                "tag":      row["tag"],
            })
    aug_df = pd.DataFrame(rows)
    full   = pd.concat(
        [df[["user_msg", "intent", "tag"]], aug_df],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[Aug] {len(df):,} → {len(full):,} rows (original + ×{factor} augmented)")
    return full

print("✅ Augmentation function ready")


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN/VAL SPLIT
# ──────────────────────────────────────────────────────────────────────────────

# Split BEFORE augmentation — val must be clean
df_train_raw, df_val = train_test_split(
    df,
    test_size    = 0.10,
    random_state = 42,
    stratify     = df["intent"],
)

print(f"Raw train : {len(df_train_raw):,}")
print(f"Val       : {len(df_val):,}")

# ── Selective augmentation — rare intents get more ──
RARE_INTENTS  = {"follow_up", "persuasion", "confirmation", "reminder"}
RARE_FACTOR   = 8    # rare intents
NORMAL_FACTOR = 4    # all others

rows = []
for _, row in df_train_raw.iterrows():
    factor = RARE_FACTOR if row["intent"] in RARE_INTENTS else NORMAL_FACTOR
    for _ in range(factor):
        rows.append({
            "user_msg": augment_text(row["user_msg"]),
            "intent":   row["intent"],
            "tag":      row["tag"],
        })

aug_df   = pd.DataFrame(rows)
df_train = pd.concat(
    [df_train_raw[["user_msg", "intent", "tag"]], aug_df],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal train : {len(df_train):,}  (selective augmentation)")
print(f"Final val   : {len(df_val):,}  (clean, no augmentation)")

print(f"\nTrain intent distribution:")
for intent, count in df_train["intent"].value_counts().items():
    pct   = count / len(df_train) * 100
    bar   = "█" * int(pct / 2)
    flag  = " ⭐ rare" if intent in RARE_INTENTS else ""
    print(f"  {intent:<30} {count:>7,}  ({pct:.1f}%)  {bar}{flag}")


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ──────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    _AR_NORM = [
        (re.compile(r"[إأآا]"), "ا"),
        (re.compile(r"[يى]"),   "ي"),
        (re.compile(r"ة"),      "ه"),
        (re.compile(r"\u0640"), ""),
        (re.compile(r"[\u064b-\u065f]"), ""),
    ]

    def __init__(self, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.w2i        = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def _normalize(self, text):
        for pattern, repl in self._AR_NORM:
            text = pattern.sub(repl, text)
        return text.lower()

    def _tokenize(self, text):
        text     = self._normalize(str(text))
        words    = re.findall(r"[\u0600-\u06ff]+|[a-z0-9][a-z0-9'\-]*", text)
        trigrams = [f"#{w[i:i+3]}" for w in words
                    for i in range(max(0, len(w) - 2))]
        return words + trigrams

    def build(self, texts):
        counter = Counter(
            tok for text in texts for tok in self._tokenize(text)
        )
        for tok, _ in counter.most_common(self.vocab_size - 4):
            if tok not in self.w2i:
                self.w2i[tok] = len(self.w2i)
        total   = len(counter)
        covered = sum(1 for t in counter if t in self.w2i)
        print(f"[Tokenizer] Vocab    : {len(self.w2i):,} tokens")
        print(f"[Tokenizer] Coverage : {covered}/{total} ({covered/total:.1%})")

    def encode(self, text):
        toks = self._tokenize(text)[: self.seq_len - 2]
        ids  = ([self.BOS]
                + [self.w2i.get(t, self.UNK) for t in toks]
                + [self.EOS])
        ids += [self.PAD] * (self.seq_len - len(ids))
        return ids[: self.seq_len]


tok = Tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
tok.build(df_train["user_msg"].tolist())
print(f"✅ Tokenizer built on {len(df_train):,} training texts")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET & DATALOADER
# ──────────────────────────────────────────────────────────────────────────────

class ConversationDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.ids     = [tokenizer.encode(t) for t in df["user_msg"].tolist()]
        self.intents = intent_encoder.transform(df["intent"].tolist())
        self.tags    = np.stack([encode_tags(t) for t in df["tag"].tolist()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ids[idx],    dtype=torch.long),
            torch.tensor(self.intents[idx],dtype=torch.long),
            torch.tensor(self.tags[idx],   dtype=torch.float32),
        )


print("Building datasets...")
train_dataset = ConversationDataset(df_train, tok)
val_dataset   = ConversationDataset(df_val,   tok)

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    pin_memory  = True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE * 2,
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
)

print(f"✅ Datasets ready")
print(f"   Train : {len(train_dataset):,}  ({len(train_loader):,} batches)")
print(f"   Val   : {len(val_dataset):,}  ({len(val_loader):,} batches)")


# ──────────────────────────────────────────────────────────────────────────────
# TEXTCNN MODEL
# ──────────────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Embedding → Conv1D(k=3,4,5) → GlobalMaxPool → Shared MLP
                                                 ├── Intent head (softmax)
                                                 └── Tags head   (sigmoid)
    """
    def __init__(self, vocab_size, n_intents, n_tags,
                 embed_dim=EMBED_DIM, filters=CNN_FILTERS, dropout=DROPOUT):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv3 = nn.Conv1d(embed_dim, filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embed_dim, filters, kernel_size=5, padding=2)

        feat_dim = 3 * filters   # 576

        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.intent_head = nn.Linear(feat_dim, n_intents)
        self.tags_head   = nn.Linear(feat_dim, n_tags)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        for conv in [self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.intent_head.weight)
        if N_TAGS > 0:
            nn.init.xavier_normal_(self.tags_head.weight)

    def forward(self, ids):
        x = self.embedding(ids)       # (B, T, D)
        x = x.permute(0, 2, 1)       # (B, D, T)

        f3 = torch.relu(self.conv3(x)).max(dim=2).values
        f4 = torch.relu(self.conv4(x)).max(dim=2).values
        f5 = torch.relu(self.conv5(x)).max(dim=2).values

        feat          = torch.cat([f3, f4, f5], dim=1)
        h             = self.shared(feat)
        intent_logits = self.intent_head(h)
        tag_logits    = self.tags_head(h)

        return intent_logits, tag_logits


model = TextCNN(
    vocab_size = len(tok.w2i),
    n_intents  = N_INTENTS,
    n_tags     = N_TAGS,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ TextCNN on {device}")
print(f"   Vocab      : {len(tok.w2i):,}")
print(f"   Intents    : {N_INTENTS}")
print(f"   Tags       : {N_TAGS}")
print(f"   Parameters : {total_params:,}")


# ──────────────────────────────────────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

intent_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
tags_criterion   = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr           = LEARNING_RATE,
    weight_decay = 1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max   = EPOCHS,
    eta_min = LEARNING_RATE * 0.01,
)

print("✅ CrossEntropyLoss (label_smoothing=0.1) + BCEWithLogitsLoss")
print("✅ Adam + CosineAnnealingLR")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE & PER-INTENT ACCURACY
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    """Returns intent accuracy and tag macro-F1."""
    model.eval()
    all_intent_preds, all_intent_true = [], []
    all_tag_preds,    all_tag_true    = [], []

    with torch.no_grad():
        for ids, intents, tags in loader:
            ids    = ids.to(device)
            il, tl = model(ids)
            preds  = il.argmax(dim=1).cpu()
            tag_p  = (torch.sigmoid(tl) > 0.35).cpu().int()
            all_intent_preds.append(preds)
            all_intent_true.append(intents)
            all_tag_preds.append(tag_p)
            all_tag_true.append(tags.int())

    ip = torch.cat(all_intent_preds).numpy()
    it = torch.cat(all_intent_true).numpy()
    tp = torch.cat(all_tag_preds).numpy()
    tt = torch.cat(all_tag_true).numpy()

    intent_acc = (ip == it).mean()

    f1s = []
    for j in range(tt.shape[1]):
        tpj = ((tp[:, j] == 1) & (tt[:, j] == 1)).sum()
        fp  = ((tp[:, j] == 1) & (tt[:, j] == 0)).sum()
        fn  = ((tp[:, j] == 0) & (tt[:, j] == 1)).sum()
        p   = tpj / max(tpj + fp, 1)
        r   = tpj / max(tpj + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))

    model.train()
    return float(intent_acc), float(np.mean(f1s)) if f1s else 0.0


def per_intent_accuracy(model, loader, name=""):
    """Prints per-intent accuracy and warns on class collapse."""
    model.eval()
    correct_per = defaultdict(int)
    total_per   = defaultdict(int)

    with torch.no_grad():
        for ids, intents, _ in loader:
            ids    = ids.to(device)
            logits, _ = model(ids)
            preds  = logits.argmax(dim=1).cpu()
            for pred, true in zip(preds.numpy(), intents.numpy()):
                total_per[true]   += 1
                correct_per[true] += int(pred == true)

    print(f"\n── Per-intent accuracy [{name}] ──")
    collapse = False
    for idx, intent in enumerate(intent_encoder.classes_):
        total = total_per.get(idx, 0)
        if total == 0:
            print(f"  {intent:<30}  — no examples")
            continue
        acc = correct_per[idx] / total
        bar = "█" * int(acc * 25)
        flag = " ⚠️" if acc < 0.5 else ""
        print(f"  {intent:<30}  {acc:.3f}  [{bar:<25}]  ({total:,}){flag}")
        if acc == 0.0 and total > 5:
            collapse = True

    if collapse:
        print("\n  ⚠️  CLASS COLLAPSE on some intents!")

    overall = sum(correct_per.values()) / max(sum(total_per.values()), 1)
    print(f"\n  Overall accuracy: {overall:.4f}")
    model.train()
    return overall

print("✅ Evaluation functions ready")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

best_val_acc   = 0.0
best_state     = None
no_improvement = 0
history        = {"loss": [], "train_acc": [], "val_acc": [], "tag_f1": []}

print("=" * 68)
print(f"  Train : {len(train_dataset):,}  |  Val : {len(val_dataset):,}")
print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Device: {device}")
print("=" * 68)
print(f"{'Epoch':>6} {'Loss':>8} {'Tr Acc':>8} {'Val Acc':>8} "
      f"{'Tag F1':>8} {'LR':>10}")
print("─" * 58)

model.train()

for epoch in range(1, EPOCHS + 1):
    epoch_loss    = 0.0
    epoch_correct = 0
    epoch_total   = 0

    for ids, intents, tags in train_loader:
        ids     = ids.to(device)
        intents = intents.to(device)
        tags    = tags.to(device)

        optimizer.zero_grad()

        intent_logits, tag_logits = model(ids)
        loss = (intent_criterion(intent_logits, intents)
                + 0.4 * tags_criterion(tag_logits, tags))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss    += loss.item()
        preds          = intent_logits.argmax(dim=1)
        epoch_correct += (preds == intents).sum().item()
        epoch_total   += len(intents)

    scheduler.step()

    # ── Evaluate every 3 epochs ──
    if epoch % 3 == 0 or epoch <= 3 or epoch == EPOCHS:
        tr_acc           = epoch_correct / max(epoch_total, 1)
        val_acc, tag_f1  = evaluate(model, val_loader)
        avg_loss         = epoch_loss / len(train_loader)
        current_lr       = scheduler.get_last_lr()[0]

        star = " ⭐" if val_acc > best_val_acc else ""
        print(f"{epoch:>6}  {avg_loss:>8.4f}  {tr_acc:>8.3f}  "
              f"{val_acc:>8.3f}  {tag_f1:>8.3f}  {current_lr:>10.2e}{star}")

        history["loss"].append(avg_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["tag_f1"].append(tag_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"\n[EarlyStop] Stopped at epoch {epoch}.")
                break

# Restore best weights
if best_state:
    model.load_state_dict({k: v.to(device)
                           for k, v in best_state.items()})

print(f"\n✅ Training complete — Best val accuracy: {best_val_acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# PER-INTENT REPORT
# ──────────────────────────────────────────────────────────────────────────────

per_intent_accuracy(model, val_loader, name="Validation — Final")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE MODEL
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state":    model.state_dict(),
    "tok":            tok,
    "intent_encoder": intent_encoder,
    "tag_vocab":      TAG_VOCAB,
    "tag_index":      TAG_INDEX,
    "intents":        INTENTS,
    "n_intents":      N_INTENTS,
    "n_tags":         N_TAGS,
    "vocab_size":     len(tok.w2i),
    "embed_dim":      EMBED_DIM,
    "cnn_filters":    CNN_FILTERS,
    "seq_len":        SEQ_LEN,
}, MODEL_PATH)

size_mb = MODEL_PATH.stat().st_size / 1e6
print(f"✅ Saved .pt → {MODEL_PATH}  ({size_mb:.2f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE MODEL AS .pkl
# ──────────────────────────────────────────────────────────────────────────────

PKL_PATH = MODEL_PATH.with_suffix(".pkl")

# Move to CPU before pickling
model.cpu()

pkl_payload = {
    "model":          model,
    "tok":            tok,
    "intent_encoder": intent_encoder,
    "tag_vocab":      TAG_VOCAB,
    "tag_index":      TAG_INDEX,
    "intents":        INTENTS,
    "n_intents":      N_INTENTS,
    "n_tags":         N_TAGS,
    "vocab_size":     len(tok.w2i),
    "embed_dim":      EMBED_DIM,
    "cnn_filters":    CNN_FILTERS,
    "seq_len":        SEQ_LEN,
}

with open(PKL_PATH, "wb") as f:
    pickle.dump(pkl_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = PKL_PATH.stat().st_size / 1e6
print(f"✅ Saved .pkl → {PKL_PATH}  ({size_mb:.2f} MB)")

# Move model back to GPU for continued use
model.to(device)
print(f"✅ Model back on {device}")


# ──────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTOR & PREDICT
# ──────────────────────────────────────────────────────────────────────────────

def detect_language(text):
    _DARIJA  = {"bghit","kifash","wash","ndir","dyali","dyal","n7awel",
                "nkhed","flous","daba","machi","bzzaf","chhal","chno",
                "ndkhol","ma9darsh","3br","khdamach","blokiha","l7sab"}
    _FRENCH  = {"je","vous","mon","ma","les","est","pas","une","des",
                "du","et","en","pour","que","qui","avec","comment",
                "voudrais","veux","compte","carte","pourquoi","quand"}
    _ENGLISH = {"i","my","the","is","are","can","do","want","help",
                "how","what","would","could","have","need","please",
                "account","bank","card","transfer","password"}
    ar = len(re.findall(r"[\u0600-\u06ff]", text))
    if ar > 0:
        latin = set(re.findall(r"[a-z0-9]+", text.lower()))
        return "darija" if latin & _DARIJA else "arabic"
    words = set(re.findall(r"[a-z']+", text.lower()))
    if words & {"bghit","kifash","dyal","machi","bzzaf","n7awel","ndkhol"}:
        return "darija"
    fr = len(words & _FRENCH) + 2 * any(c in text for c in "àâéèêëîïôùûüç")
    en = len(words & _ENGLISH)
    return "french" if fr >= en else "english"


def predict(text, tag_threshold=0.35):
    model.eval()
    ids = torch.tensor([tok.encode(text)], dtype=torch.long).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        intent_logits, tag_logits = model(ids)
    ms = (time.perf_counter() - t0) * 1000

    # ── Intent ──
    intent_proba = torch.softmax(intent_logits[0], dim=0).cpu().numpy()
    best_idx     = int(intent_proba.argmax())
    intent       = intent_encoder.classes_[best_idx]
    conf         = float(intent_proba[best_idx])

    # ── Tags: model head + keyword map merged ──
    tag_probs       = torch.sigmoid(tag_logits).squeeze(0).cpu().numpy()
    model_tags      = [TAG_VOCAB[i] for i, p in enumerate(tag_probs) if p > tag_threshold]
    keyword_tag_str = enrich_tags_from_text(text, "")
    keyword_tags    = [t.strip() for t in keyword_tag_str.split(",")
                       if t.strip() not in ("", "untagged")]
    merged          = list(dict.fromkeys(model_tags + keyword_tags))
    pred_tags       = resolve_conflicts(merged)

    # ── Language ──
    lang = detect_language(text)

    # ── Pretty print ──
    W = 70
    print("─" * W)
    print(f"  💬 {text}")
    print("─" * W)
    print(f"  INTENT  : {intent.upper()}")
    print(f"  CONF    : {conf:.0%}")
    print(f"  LANG    : {lang}")
    print(f"  TAGS    : {', '.join(pred_tags) if pred_tags else '—'}")
    print(f"  TIME    : {ms:.1f} ms")
    print("─" * W)


    return {
        "intent":       intent,
        "confidence":   round(conf, 4),
        "tags":         pred_tags,
        "lang":         lang,
        "inference_ms": round(ms, 2),
        "all_intents":  {
            intent_encoder.classes_[i]: round(float(p), 4)
            for i, p in enumerate(intent_proba)
        },
    }

print("✅ predict() ready — usage: predict('your message here')")


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK & INFERENCE TESTS  (only runs when executed directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model.eval()
    test_ids = torch.tensor(
        [tok.encode("bghit n3ref aktar 3la had l offre")],
        dtype=torch.long
    ).to(device)

    for _ in range(50):   # warmup
        with torch.no_grad():
            model(test_ids)

    N  = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            model(test_ids)
    ms = (time.perf_counter() - t0) / N * 1000

    print(f"[Benchmark] {ms:.3f} ms per inference ({N} runs on {device})")
    print(f"[Benchmark] {1000/ms:.0f} messages/second")


    # ──────────────────────────────────────────────────────────────────────────────
    # INFERENCE TESTS
    # ──────────────────────────────────────────────────────────────────────────────

    predict("Can you send me my account statement for this month?")
    predict("Please block my card immediately")
    predict("je voudrais réinitialiser mon mot de passe")
    predict("pouvez-vous m'envoyer mon relevé de compte?")
    predict("bghit n7awel flous l compte dyal khoya")
    predict("3tini RIB dyal compte dyali")
    predict("أريد إيقاف بطاقتي فوراً")
    predict("أرسل لي كشف حسابي من فضلك")
    predict("I am interested in learning more about your loan products")
    predict("Tell me more about your savings account options")
    predict("What investment plans do you currently offer?")
    predict("je suis intéressé par vos offres de crédit immobilier")
    predict("dites-moi plus sur vos nouveaux produits")
    predict("bghit n3ref aktar 3la l offres dyal lbank")
    predict("أنا مهتم بمعرفة المزيد عن منتجات التوفير لديكم")
    predict("ما هي عروض القروض المتاحة عندكم؟")
    predict("How are you doing today?")
    predict("Nice to chat with you!")
    predict("You are very helpful, I appreciate it")
    predict("comment tu vas aujourd'hui?")
    predict("c'est sympa de parler avec toi")
    predict("labas 3lik? kif dayir?")
    predict("كيف حالك اليوم؟")
    predict("أنت مفيد جداً شكراً لك")
    predict("Your new app design is much better than before, well done")
    predict("The waiting time at the branch is way too long")
    predict("I love how fast the transfers work now")
    predict("votre nouvelle application est vraiment bien faite")
    predict("le service en agence laisse vraiment à désirer")
    predict("l app jdida mzyana bzzaf, 3jbatni")
    predict("التطبيق الجديد رائع، أحسنتم كثيراً")
    predict("أعتقد أن الخدمة تحتاج إلى تحسين كبير")
    predict("Can you waive the annual fee for my card this year?")
    predict("I have been a client for 10 years, can I get a better rate?")
    predict("Is there any way to reduce the transfer fees?")
    predict("pouvez-vous me faire un geste sur les frais de dossier?")
    predict("je suis client depuis longtemps, j'aimerais un meilleur taux")
    predict("wash ymken tkhfiw liya frais dyal lcompte?")
    predict("هل يمكنكم تخفيض رسوم التحويل لي؟")
    predict("أنا عميل قديم، هل يمكنني الحصول على سعر فائدة أفضل؟")
    predict("I just want to confirm my transfer went through successfully")
    predict("Can you confirm my appointment for tomorrow at 10am?")
    predict("Is my new card already activated?")
    predict("pouvez-vous confirmer que mon virement a bien été effectué?")
    predict("je veux juste vérifier que mon rendez-vous est confirmé")
    predict("bghit n2akad ila l transfert dyali wsal mzyan")
    predict("هل يمكنكم تأكيد وصول حوالتي؟")
    predict("أريد التأكد من أن موعدي غداً مؤكد")
    predict("I am absolutely furious about what happened to my account")
    predict("I feel so stressed, I don't know what to do anymore")
    predict("This situation is making me really anxious about my money")
    predict("je suis vraiment paniqué, je ne sais plus quoi faire")
    predict("je me sens complètement abandonné par votre banque")
    predict("ana mherres bzzaf, had lmochkil kaydi3ni")
    predict("أنا في حالة ذعر تام، لا أعرف ماذا أفعل")
    predict("أشعر بالقلق الشديد بشأن أموالي")
    predict("I am reminding you that my issue from last week is not resolved")
    predict("This is my third time contacting you about the same problem")
    predict("I sent a complaint 5 days ago and heard nothing back")
    predict("je vous rappelle que mon problème n'est toujours pas réglé")
    predict("c'est la deuxième fois que je vous contacte pour la même chose")
    predict("hada marra tanya kankllmkom 3la nfs lmochkil")
    predict("أذكركم بأن مشكلتي لم تُحل حتى الآن")
    predict("هذه المرة الثالثة التي أتصل فيها بشأن نفس الأمر")
    predict("Yes I understand, thank you for explaining that")
    predict("OK got it, I will do that right away")
    predict("Perfect, that answers my question completely")
    predict("oui je comprends, merci pour l'explication")
    predict("ok c'est bon, j'ai bien compris merci")
    predict("wakha fhmt, choukran 3la chi7")
    predict("نعم فهمت، شكراً على التوضيح")
    predict("حسناً، استوعبت ذلك تماماً")
    predict("You really should fix this before I post about it on social media")
    predict("If you help me today I will recommend your bank to all my friends")
    predict("It is in your best interest to resolve this quickly")
    predict("si vous ne réglez pas ça je vais sur les réseaux sociaux")
    predict("si vous m'aidez maintenant je vous recommande à tout le monde")
    predict("ila ma7lawtich had lmochkil ghadi nktb 3liha f facebook")
    predict("إذا لم تحلوا مشكلتي سأنشر تجربتي على وسائل التواصل")
    predict("إذا ساعدتموني الآن سأوصي بكم لجميع أصدقائي")
    predict("Hello, good morning!")
    predict("Hi there, hope you are well")
    predict("Goodbye, have a great day")
    predict("Bonjour, comment puis-je vous aider?")
    predict("Au revoir et bonne journée!")
    predict("Salam, labas 3lik?")
    predict("Bessaha, bslama!")
    predict("مرحباً، صباح الخير")
    predict("وداعاً، أتمنى لكم يوماً طيباً")
    predict("I called yesterday about my blocked card, what is the update?")
    predict("Any news on my refund request from last week?")
    predict("I submitted my documents 3 days ago, what is the status?")
    predict("j'avais signalé un problème hier, qu'est-ce qui a été fait?")
    predict("des nouvelles concernant ma demande de remboursement?")
    predict("bghit n3ref ach wqe3 f dossier dyali li sddt lbara7")
    predict("أتصل بخصوص الشكوى التي قدمتها الأسبوع الماضي، ما الجديد؟")
    predict("هل هناك أي تحديث بشأن طلب استرداد أموالي؟")
    predict("help")
    predict("merci")
    predict("ok")
    predict("شكراً")
    predict("bghit")
    predict("I want to n7awel flous but the app ma khdamach")
    predict("bghit transfer money l compte dyal mon frère")
    predict("j'ai oublié mon PIN و ما قدرتش دخول للحساب")
    predict("I am vraiment frustrated, l app ma khdamach")
    predict("I already told you about this problem")
    predict("What happened to my request from yesterday?")
    predict("I think you should do better")
    predict("Please just fix it")
    predict("I understand, thanks")
    predict("I need you to fix this problem using ai")
    predict("you should go to marrakesh")
