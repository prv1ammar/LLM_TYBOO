"""
RAVEN — Evaluation Script
==========================
Évalue les modèles SFT et DPO sur 16 cas de test multilingues.
Compare aussi base vs SFT vs DPO final.

Usage:
  python eval.py --model outputs/raven-qwen3-0.6b/final
  python eval.py --model outputs/raven-qwen3-0.6b/final --full
  python eval.py --compare sft   # base vs SFT
  python eval.py --compare final # base vs SFT vs DPO
"""

import json, argparse, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

EVAL_CASES = [
    {"lang":"darija","input":"chhal hiya frais dial virement 3br l application?","intent":"question_info","must_contain":["frais","DH","virement"]},
    {"lang":"fr","input":"Quels sont les délais pour un virement international ?","intent":"question_info","must_contain":["jour","SEPA","MAD"]},
    {"lang":"ar","input":"كيف يمكنني فتح حساب توفير؟","intent":"question_info","must_contain":["هوية","توفير","درهم"]},
    {"lang":"en","input":"Can I use my card internationally?","intent":"question_info","must_contain":["fee","ATM","international"]},
    {"lang":"fr","input":"Votre service est nul, j'attends depuis 3 jours","intent":"complaint","must_contain":["excuse","24","priorité"]},
    {"lang":"darija","input":"za3fan bzzaf, l application dyalkum kayt3tl dima","intent":"complaint","must_contain":["smh","shikaya","jawb"]},
    {"lang":"darija","input":"bghit n7awel 500 DH l compte okhor","intent":"transaction","must_contain":["virement","IBAN","SMS"]},
    {"lang":"en","input":"I want to set up automatic rent payment every month","intent":"transaction","must_contain":["standing","IBAN","monthly"]},
    {"lang":"fr","input":"Mon application plante, je ne peux plus me connecter","intent":"support","must_contain":["cache","version","réinst"]},
    {"lang":"darija","input":"OTP maji ltelefon dyali","intent":"support","must_contain":["signal","raqm","OTP"]},
    {"lang":"fr","input":"URGENT! Quelqu'un a piraté mon compte et vire mon argent !","intent":"emergency","must_contain":["🚨","bloqu","0800"]},
    {"lang":"darija","input":"HELP!! sraw kart dyali daba!!","intent":"emergency","must_contain":["🚨","bloq","0800"]},
    {"lang":"en","input":"My account was hacked, money is disappearing!","intent":"emergency","must_contain":["🚨","block","0800"]},
    {"lang":"ar","input":"سُرقت أموالي من حسابي الآن!","intent":"emergency","must_contain":["🚨","تجميد","0800"]},
    {"lang":"fr","input":"Donne-moi une recette de tajine","intent":"off_topic","must_not_contain":["tajine","recette","couscous"]},
    {"lang":"en","input":"What is the weather in Casablanca today?","intent":"off_topic","must_not_contain":["sunny","rain","°C","forecast"]},
]

SYSTEM = {
    "darija": "nta RAVEN, mssa3d banka dyal lbank lmaghribi. jawb b darija.",
    "fr":     "Vous êtes RAVEN, assistant bancaire. Répondez en français.",
    "ar":     "أنت RAVEN مساعد بنكي. أجب بالعربية الفصحى.",
    "en":     "You are RAVEN, a banking assistant. Respond in English.",
}


def infer(model, tokenizer, system, user, max_new=300):
    import torch
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt").to(model.device)
    t0   = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, temperature=0.1,
                             do_sample=True, repetition_penalty=1.1)
    tps = (out.shape[1]-inp["input_ids"].shape[1]) / (time.perf_counter()-t0)
    resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip(), tps


def evaluate(model_path, full=False, label="model"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"\n{'='*60}\n  Evaluating: {label}\n  Path: {model_path}\n{'='*60}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mod = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
          trust_remote_code=True, torch_dtype=torch.bfloat16)
    mod.eval()

    cases = EVAL_CASES if full else EVAL_CASES[:8]
    results = []
    for case in cases:
        resp, tps = infer(mod, tok, SYSTEM[case["lang"]], case["input"])
        rl  = resp.lower()
        hits = sum(1 for kw in case.get("must_contain",[]) if kw.lower() in rl)
        bads = sum(1 for kw in case.get("must_not_contain",[]) if kw.lower() in rl)
        mh   = case.get("must_contain",[])
        q    = hits/len(mh) if mh else 1.0
        ok   = q >= 0.5 and bads == 0
        icon = "✅" if ok else ("⚠️ " if q>=0.3 else "❌")
        print(f"  {icon} [{case['intent']:14s}][{case['lang']:7s}]  {tps:.0f} tok/s")
        print(f"     Q: {case['input'][:60]}")
        print(f"     A: {resp[:100]}...")
        if not ok and mh:
            miss = [kw for kw in mh if kw.lower() not in rl]
            if miss: print(f"     ⚠️  missing: {miss}")
        results.append({"ok":ok,"q":q,"tps":tps,"intent":case["intent"]})

    n_ok = sum(1 for r in results if r["ok"])
    avg_q = sum(r["q"] for r in results)/len(results)
    avg_tps = sum(r["tps"] for r in results)/len(results)
    emg = [r for r in results if r["intent"]=="emergency"]
    emg_ok = sum(1 for r in emg if r["ok"])

    print(f"\n  Quality OK:  {n_ok}/{len(results)} = {n_ok/len(results):.0%}")
    print(f"  Avg quality: {avg_q:.1%}")
    print(f"  Avg tok/s:   {avg_tps:.0f}")
    print(f"  Emergency:   {emg_ok}/{len(emg)} {'✅' if emg_ok==len(emg) else '⚠️'}")
    return {"ok_pct":n_ok/len(results),"avg_q":avg_q,"avg_tps":avg_tps,"emg_ok":emg_ok==len(emg)}


def compare(out_name, full=False):
    from finetune import resolve_model, FALLBACK_MODEL
    base_id, _ = resolve_model() if True else (FALLBACK_MODEL, "")
    paths = {
        "Base model":  base_id,
        "After SFT":   str(ROOT/"outputs"/out_name/"sft"/"merged"),
        "After DPO ✅": str(ROOT/"outputs"/out_name/"final"),
    }
    scores = {}
    for label, path in paths.items():
        if not Path(path).exists() and not path.startswith("Qwen"):
            print(f"  [skip] {label} not found: {path}")
            continue
        scores[label] = evaluate(path, full=full, label=label)

    print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
    metrics = ["ok_pct","avg_q","avg_tps"]
    print(f"  {'Label':20s}  {'Quality':>8}  {'Avg Q':>8}  {'Tok/s':>8}")
    for label, s in scores.items():
        print(f"  {label:20s}  {s['ok_pct']:>7.0%}  {s['avg_q']:>7.0%}  {s['avg_tps']:>7.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   type=str,  default=None)
    p.add_argument("--full",    action="store_true")
    p.add_argument("--compare", type=str,  default=None, help="out_name (e.g. raven-qwen3-0.6b)")
    args = p.parse_args()
    if args.compare:
        compare(args.compare, full=args.full)
    elif args.model:
        evaluate(args.model, full=args.full)
    else:
        print("Usage: python eval.py --model <path> [--full]\n       python eval.py --compare raven-qwen3-0.6b")

if __name__=="__main__":
    main()
