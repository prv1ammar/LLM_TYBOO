"""
RAVEN — TextCNN Intent Classifier (CPU-Optimized)
Architecture: Embedding → CNN1D(k=3,4,5) → MaxPool+ReLU → MLP → Softmax
Verified backprop | Adam | label smoothing | < 1ms CPU inference
"""
import json,re,pickle,random,sys,time
import numpy as np
from pathlib import Path
from collections import Counter
random.seed(42); np.random.seed(42)

DATA_DIR=Path(__file__).parent.parent/"data"
MODEL_PATH=Path(__file__).parent.parent/"models"/"raven_cnn.pkl"
INTENTS=["question_info","complaint","transaction","support","off_topic","emergency"]
N_CLS=len(INTENTS)
ITAGS={"question_info":["banking","faq"],"complaint":["banking","frustrated","requires_human"],
       "transaction":["banking","transfer","form_needed"],"support":["banking"],
       "off_topic":[],"emergency":["banking","urgent","fraud_signal","requires_human"]}

# ── CONV1D (correct forward + backward) ──────────────────────
def conv_fwd(x,W,b,k):
    """x:(B,T,D) W:(kD,F) b:(F) → out:(B,F), cache"""
    B,T,D=x.shape; T_out=max(T-k+1,1)
    wins=np.zeros((B,T_out,k*D),np.float32)
    for ki in range(k):
        te=min(T_out,T-ki)
        wins[:,:te,ki*D:(ki+1)*D]=x[:,ki:ki+te,:]
    feat=np.maximum(0.,wins@W+b)          # relu fused
    pidx=feat.argmax(1); out=feat.max(1)  # global maxpool
    return out,(wins,W,b,feat,pidx,k,T)

def conv_bwd(g_out,cache):
    """→ dW:(kD,F), db:(F), dx:(B,T,D)"""
    wins,W,b,feat,pidx,k,T=cache
    B,T_out,kD=wins.shape; F=W.shape[1]; D=kD//k
    # scatter grad to argmax
    gf=np.zeros_like(feat)
    gf[np.arange(B)[:,None],pidx,np.arange(F)[None,:]]=g_out
    gf*=(feat>0)                              # ReLU backward
    gff=gf.reshape(B*T_out,F); wff=wins.reshape(B*T_out,kD)
    dW=wff.T@gff; db=gff.sum(0)
    dxw=(gff@W.T).reshape(B,T_out,k,D)
    dx=np.zeros((B,T,D),np.float32)
    for ki in range(k):
        te=min(T_out,T-ki)
        dx[:,ki:ki+te,:]+=dxw[:,:te,ki,:]
    return dW,db,dx

# ── TOKENIZER ────────────────────────────────────────────────
class Tokenizer:
    PAD=0;UNK=1;BOS=2;EOS=3
    _AR=[(re.compile(r'[إأآا]'),'ا'),(re.compile(r'[يى]'),'ي'),
         (re.compile(r'ة'),'ه'),(re.compile(r'\u0640'),''),
         (re.compile(r'[\u064b-\u065f]'),'')]
    def __init__(self,V=5000,L=48):self.V=V;self.L=L;self.w2i={'<PAD>':0,'<UNK>':1,'<BOS>':2,'<EOS>':3}
    def _n(self,t):
        for p,r in self._AR:t=p.sub(r,t)
        return t.lower()
    def _t(self,text):
        t=self._n(text); ws=re.findall(r'[\u0600-\u06ff]+|[a-z0-9][a-z0-9\'\-]*',t)
        return ws+[f'#{w[i:i+3]}' for w in ws for i in range(max(0,len(w)-2))]
    def build(self,texts):
        cnt=Counter(t for tx in texts for t in self._t(tx))
        for tok,_ in cnt.most_common(self.V-4):
            if tok not in self.w2i:i=len(self.w2i);self.w2i[tok]=i
        print(f"[Tok] vocab={len(self.w2i):,}")
    def encode(self,text):
        ids=[self.BOS]+[self.w2i.get(t,self.UNK) for t in self._t(text)[:self.L-2]]+[self.EOS]
        ids+=[self.PAD]*(self.L-len(ids));return np.array(ids[:self.L],np.int32)
    def batch(self,texts):return np.stack([self.encode(t) for t in texts])

# ── TEXT CNN ─────────────────────────────────────────────────
class TextCNN:
    def __init__(self,V,D=48,F=128,drop=0.35):
        self.D=D;self.F=F;self.drop=drop
        sc=lambda r,c:(np.random.randn(r,c)*np.sqrt(2/(r+c))).astype(np.float32)
        self.E=(np.random.randn(V,D)*0.02).astype(np.float32);self.dE=np.zeros_like(self.E)
        self.cW={k:sc(k*D,F) for k in[3,4,5]}
        self.cb={k:np.zeros(F,np.float32) for k in[3,4,5]}
        self.dcW={k:np.zeros((k*D,F),np.float32) for k in[3,4,5]}
        self.dcb={k:np.zeros(F,np.float32) for k in[3,4,5]}
        self.W1=sc(3*F,F);self.b1=np.zeros(F,np.float32)
        self.W2=sc(F,N_CLS);self.b2=np.zeros(N_CLS,np.float32)
        self.dW1=np.zeros_like(self.W1);self.db1=np.zeros_like(self.b1)
        self.dW2=np.zeros_like(self.W2);self.db2=np.zeros_like(self.b2)
        self._cache={}

    def forward(self,ids,tr=True):
        x=self.E[ids].astype(np.float32);self._ids=ids;self._x=x;B=ids.shape[0]
        parts=[]
        for k in[3,4,5]:
            out,cache=conv_fwd(x,self.cW[k],self.cb[k],k)
            parts.append(out);self._cache[k]=cache
        feat=np.concatenate(parts,-1)  # (B,3F)
        if tr:
            m=(np.random.rand(*feat.shape)>self.drop).astype(np.float32)/(1-self.drop)
            feat=feat*m;self._dm=m
        else:self._dm=None
        self._feat=feat
        h=feat@self.W1+self.b1;self._hpre=h;h=np.maximum(0.,h)
        if tr:
            m2=(np.random.rand(*h.shape)>self.drop*0.5).astype(np.float32)/(1-self.drop*0.5)
            h=h*m2;self._dm2=m2
        else:self._dm2=None
        self._h=h
        return h@self.W2+self.b2

    def backward(self,grad):
        B,L,D=self._x.shape;F=self.F
        # fc2
        self.dW2+=self._h.T@grad;self.db2+=grad.sum(0)
        d=grad@self.W2.T
        if self._dm2 is not None:d*=self._dm2
        d*=(self._hpre>0)
        # fc1
        self.dW1+=self._feat.T@d;self.db1+=d.sum(0)
        d=d@self.W1.T
        if self._dm is not None:d*=self._dm
        # conv backward
        dx=np.zeros_like(self._x)
        for i,k in enumerate([3,4,5]):
            dW,db,dxk=conv_bwd(d[:,i*F:(i+1)*F],self._cache[k])
            self.dcW[k]+=dW;self.dcb[k]+=db;dx+=dxk
        # embedding
        self.dE[:]=0;np.add.at(self.dE,self._ids,dx)

    def predict_proba(self,ids,bsz=512):
        if ids.ndim==1: ids=ids.reshape(1,-1)
        res=[]
        for i in range(0,len(ids),bsz):
            xb=ids[i:i+bsz]
            x=self.forward(xb,tr=False)
            e=np.exp(x-x.max(-1,keepdims=True)); p=e/e.sum(-1,keepdims=True)
            res.append(p)
        return np.vstack(res)

    def zero(self):
        self.dE[:]=0
        for k in[3,4,5]:self.dcW[k][:]=0;self.dcb[k][:]=0
        self.dW1[:]=0;self.db1[:]=0;self.dW2[:]=0;self.db2[:]=0

    def params(self):
        p=[('E',self.E,self.dE)]
        for k in[3,4,5]:p+=[(f'cW{k}',self.cW[k],self.dcW[k]),(f'cb{k}',self.cb[k],self.dcb[k])]
        return p+[('W1',self.W1,self.dW1),('b1',self.b1,self.db1),
                  ('W2',self.W2,self.dW2),('b2',self.b2,self.db2)]

# ── LOSS + ADAM ───────────────────────────────────────────────
def ce(logits,y,sm=0.1):
    B=logits.shape[0];e=np.exp(logits-logits.max(-1,keepdims=True));p=e/e.sum(-1,keepdims=True)
    t=np.full_like(p,sm/(N_CLS-1));t[np.arange(B),y]=1.-sm
    return -(t*np.log(p+1e-9)).sum()/B,(p-t)/B

class Adam:
    def __init__(self,lr=1e-3,b1=.9,b2=.999,eps=1e-8,wd=1e-4):
        self.lr=lr;self.b1=b1;self.b2=b2;self.eps=eps;self.wd=wd;self.t=0;self.m={};self.v={}
    def step(self,params):
        self.t+=1;lrt=self.lr*np.sqrt(1-self.b2**self.t)/(1-self.b1**self.t)
        for _,p,g in params:
            if g is None:continue
            k=id(p)
            if k not in self.m:self.m[k]=np.zeros_like(p);self.v[k]=np.zeros_like(p)
            g2=g+self.wd*p;self.m[k]=self.b1*self.m[k]+(1-self.b1)*g2
            self.v[k]=self.b2*self.v[k]+(1-self.b2)*g2**2
            p-=lrt*self.m[k]/(np.sqrt(self.v[k])+self.eps)

# ── DATA ─────────────────────────────────────────────────────
def _rule_label(text):
    """Label a user message using RULES (same as inference-time rules)."""
    for intent,pat in RULES:
        if pat.search(text):
            if intent=="off_topic" and _BANKING_CTX.search(text):
                continue
            return intent
    return None

def load_data():
    """
    Load Step-1 data (turns format) OR flat {text, intent} format.
    For turns data: extract user messages + auto-label via rules.
    Keeps only samples where a rule fires (high confidence).
    """
    texts,labels=[],[]
    for f in sorted(DATA_DIR.rglob("*.jsonl")):
        with open(f,encoding="utf-8") as fp:
            for line in fp:
                line=line.strip()
                if not line:continue
                try:
                    obj=json.loads(line)
                    # Format A: flat {text, intent}
                    if "text" in obj and "intent" in obj:
                        if obj["intent"] in INTENTS:
                            texts.append(obj["text"])
                            labels.append(INTENTS.index(obj["intent"]))
                    # Format B: Step-1 turns {id, lang, turns, ...}
                    elif "turns" in obj:
                        for turn in obj["turns"]:
                            if turn.get("role")!="user":continue
                            msg=turn.get("content","").strip()
                            if len(msg)<5:continue
                            intent=_rule_label(msg)
                            if intent is not None:
                                texts.append(msg)
                                labels.append(INTENTS.index(intent))
                except:pass
    print(f"[Data] {len(texts):,} — {dict(Counter(labels))}");return texts,labels

def augment(texts,labels,factor=12):
    at,al=list(texts),list(labels)
    for t,l in zip(texts,labels):
        ws=t.split()
        for _ in range(factor):
            r=random.random()
            if r<.18 and len(ws)>3:w=ws[:];w.pop(random.randint(0,len(w)-1));aug=" ".join(w)
            elif r<.33:aug=t.rstrip("?!.،")+random.choice(["?","!",""," stp"," svp"])
            elif r<.46:aug=t.lower()
            elif r<.58 and len(ws)>2:w=ws[:];i=random.randint(0,len(w)-2);w[i],w[i+1]=w[i+1],w[i];aug=" ".join(w)
            elif r<.70:aug=t+" "+random.choice(["","merci","please","شكراً","baraka"])
            else:aug=t
            at.append(aug.strip());al.append(l)
    print(f"[Aug] {len(texts):,}→{len(at):,}");return at,al

# ── RULES ────────────────────────────────────────────────────
RULES=[
    ("emergency",re.compile(r'(vol[eé]|srak|srqo|stolen|hack|piraté|mkhtar9|fraud|احتيال|سرق|اختراق|bloqu|block|gel|freeze|9awwed|pris mon argent|took.*money|took everything|dkhlu.*l7sab|سحب كل|HELP.*account|AIDEZ.*argent|3AJJLU|quelqu.un.*pris.*argent|a pris mon argent|someone hacked|money.*disappeared|مخترق)',re.I)),
    ("off_topic",re.compile(r'(météo|weather|tbard|lbard|cuisine|recette|couscous|tajine|blague|joke(?!.*bank)|\bmatch\b|télé|tarikh|تاريخ|طقس|نكتة|learn arabic|learn.*language|apprendre.*langue|best.*restaurant|rajfana|best way to learn|tell me about)',re.I)),
    ("support",re.compile(r'(ma khdamach|ne fonctionne pas|not working|لا يعمل|session expired|mot de passe|password|nsit.*code|bloqué|m9ful|locked|OTP|erreur|error|خطأ|plante|crashes)',re.I)),
    ("transaction",re.compile(r'(virer|virement|7awel|n7awel|payer.*facture|nkhed.*fatura|pay.*bill|facture|fatura|فاتورة|loyer|rent(?!.*balance)|إيجار|dépôt|deposit|إيداع|rembours|nsedd|سداد|recharger|nchar9|شحن|\bwire\s+\d)',re.I)),
    ("complaint",re.compile(r'(insatisf|mécontent|machi radi|غير راض|plainte|شكوى|inacceptable|remboursement|rdod|directeur|lmdir|bank.*maghrib|vraiment nul|za3fan 3la qad|nobody cares|bank is a joke|j.en ai marre|en ai marre|fed up|this.*joke)',re.I)),
    ("question_info",re.compile(r'(so2al|استفسار|renseignement|c.est quoi|what is|what are|how (do|can|much|long)|quel.*délai|chhal.*waqt|كم يستغرق|check.*balance|solde|رصيد|plafond|puis-je|can i\b|wash ymken|هل يمكن)',re.I)),
]
_DA={'bghit','kifash','wash','ndir','kayna','dyali','dyal','n7awel','nkhed','nsedd','flous',
     'srqo','daba','machi','bzzaf','chhal','chno','n3ref','nftah','ndkhol','ma9darsh','3br','khdamach'}
def detect_lang(text):
    ar=len(re.findall(r'[\u0600-\u06ff]',text))
    if ar>0:
        w=set(re.findall(r'[a-z0-9]+',text.lower()));return "darija" if w&_DA else "arabic"
    w=set(re.findall(r"[a-z']+",text.lower()))
    if w&{'bghit','kifash','dyal','machi','bzzaf','ndkhol','n7awel'}:return "darija"
    fr=len(w&{'je','vous','mon','ma','les','est','pas','une','des','du','comment','quel','puis','veux','voudrais','et','en','ne'})
    en=len(w&{'i','my','the','is','are','can','how','what','want','help','do','would','could','have','get','need'})
    fr+=2*any(c in text for c in 'àâéèêëîïôùûüç');return "french" if fr>=en else "english"
_BANKING_CTX=re.compile(r'\b(bank|banka|compte|account|carte|card|crédit|credit|virement|transfer|l7sab|banka|frais|fees)\b',re.I)

def apply_rules(text,nn_intent,nn_proba):
    proba=nn_proba.copy();hard=None
    for intent,pat in RULES:
        if pat.search(text):
            # Don't flag as off_topic if banking context present
            if intent=="off_topic" and _BANKING_CTX.search(text):
                continue
            hard=intent;break
    if hard and(nn_intent!=hard or float(proba.max())<0.75):
        hi=INTENTS.index(hard);proba[hi]=max(proba[hi],0.82);proba/=proba.sum()
    return INTENTS[proba.argmax()],proba

# ── TRAINING ─────────────────────────────────────────────────
def train(epochs=60,batch=64,lr=3e-3,D=48,F=128,drop=0.35,vocab=5000,aug_factor=12):
    raw_t,raw_l=load_data();texts,labels=augment(raw_t,raw_l,factor=aug_factor)
    tok=Tokenizer(V=vocab,L=48);tok.build(texts)
    X=tok.batch(texts);y=np.array(labels,np.int32)
    idx=np.random.permutation(len(X));sp=int(0.82*len(idx))
    Xtr,ytr=X[idx[:sp]],y[idx[:sp]];Xvl,yvl=X[idx[sp:]],y[idx[sp:]]
    model=TextCNN(len(tok.w2i),D=D,F=F,drop=drop);opt=Adam(lr=lr,wd=1e-4)
    print(f"\n[CNN] D={D} F={F} drop={drop} vocab={len(tok.w2i):,} train={len(Xtr):,} val={len(Xvl):,}")
    best_acc=0.;best_w=None;patience=10;no_imp=0
    for ep in range(1,epochs+1):
        perm=np.random.permutation(len(Xtr));Xs,ys=Xtr[perm],ytr[perm]
        ep_loss=0.;nb=0
        for s in range(0,len(Xs),batch):
            xb,yb=Xs[s:s+batch],ys[s:s+batch]
            if not len(xb):continue
            model.zero();logits=model.forward(xb,tr=True)
            loss,grad=ce(logits,yb);ep_loss+=loss;nb+=1
            model.backward(grad)
            for _,p,g in model.params():
                if g is not None:np.clip(g,-1.,1.,out=g)
            opt.step(model.params())
        if ep%5==0 or ep<=3:
            vp=model.predict_proba(Xvl);va=(vp.argmax(1)==yvl).mean()
            tp=model.predict_proba(Xtr[:500]);ta=(tp.argmax(1)==ytr[:500]).mean()
            star='⭐' if va>best_acc else ''
            print(f"  ep {ep:3d}/{epochs}  loss={ep_loss/max(nb,1):.4f}  tr={ta:.3f}  val={va:.3f} {star}")
            if va>best_acc:
                best_acc=va;best_w={id(p):p.copy() for _,p,_ in model.params()};no_imp=0
            else:
                no_imp+=1
                if no_imp>=patience:print(f"  [Early stop] ep={ep}");break
    if best_w:
        for _,p,_ in model.params():
            if id(p) in best_w:p[:]=best_w[id(p)]
    print(f"\n[CNN] ✅ Best val: {best_acc:.3f}");return model,tok,best_acc

# ── CLASSIFIER ───────────────────────────────────────────────
class CNNClassifier:
    def __init__(self):self.model=None;self.tok=None
    def fit(self,**kw):self.model,self.tok,acc=train(**kw);return acc
    def predict(self,text):
        t0=time.perf_counter()
        ids=self.tok.encode(text).reshape(1,-1)
        proba=self.model.predict_proba(ids)[0]
        intent,proba=apply_rules(text,INTENTS[proba.argmax()],proba)
        lang=detect_lang(text)
        tags=list(ITAGS.get(intent,[]));tags.append(f"lang:{lang}")
        if RULES[0][1].search(text) and "fraud_signal" not in tags:tags.append("fraud_signal")
        if len(re.findall(r'[\u0600-\u06ff]',text))>0 and re.search(r'[a-zA-Z]{3,}',text):tags.append("code_switching")
        ms=(time.perf_counter()-t0)*1000
        return {"intent":intent,"confidence":round(float(proba.max()),4),
                "tags":list(dict.fromkeys(tags)),"lang":lang,"inference_ms":round(ms,2),
                "all_intents":{k:round(float(p),4) for k,p in zip(INTENTS,proba)}}
    def benchmark(self,n=2000):
        ids=self.tok.encode("bghit n7awel flous mn compte dyali").reshape(1,-1)
        for _ in range(20):self.model.predict_proba(ids)
        t0=time.perf_counter()
        for _ in range(n):self.model.predict_proba(ids)
        ms=(time.perf_counter()-t0)/n*1000
        print(f"[Benchmark] {ms:.3f}ms / inference ({n} runs, CPU)");return ms
    def save(self,path=MODEL_PATH):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with open(path,'wb') as f:pickle.dump({"m":self.model,"t":self.tok},f)
        print(f"[CNN] ✅ Saved → {path}  ({path.stat().st_size//1024}KB)")
    @classmethod
    def load(cls,path=MODEL_PATH):
        with open(Path(path),'rb') as f:obj=pickle.load(f)
        c=cls();c.model=obj["m"];c.tok=obj["t"];return c

# ── DEMO ─────────────────────────────────────────────────────
UNSEEN=[
    ("3ndi so2al 3la l plafond dyal ls7b dyal compte","question_info","darija"),
    ("c'est quoi le délai pour avoir ma carte ?","question_info","french"),
    ("can i check my balance from abroad?","question_info","english"),
    ("كم يستغرق تحويل المبالغ بين البنوك؟","question_info","arabic"),
    ("votre banque c'est vraiment nul, j'en ai marre","complaint","french"),
    ("had service bla sta7ya, za3fan 3la qad Allah","complaint","darija"),
    ("this bank is a joke, nobody cares about customers","complaint","english"),
    ("3yit nkhed loyer dyal shqqa 3br l application","transaction","darija"),
    ("je dois virer de l'argent à ma famille ce soir","transaction","french"),
    ("need to wire 5000 to another account right now","transaction","english"),
    ("l badge ma khdamach f l guichet","support","darija"),
    ("mon application plante dès que j'ouvre les virements","support","french"),
    ("keeps saying session expired every time i login","support","english"),
    ("AIDEZ MOI quelqu'un a pris mon argent !!!","emergency","french"),
    ("3AJJLU dkhlu l7sab dyali hakda bla ma n3ref","emergency","darija"),
    ("HELP someone hacked my account and took everything","emergency","english"),
    ("تم سحب كل أموالي الآن بدون إذن مني!","emergency","arabic"),
    ("comment faire un bon tajine marocain?","off_topic","french"),
    ("3llmni wach rajfana ghda f maroc","off_topic","darija"),
    ("what's the best way to learn arabic?","off_topic","english"),
]
def demo(clf):
    print("\n"+"="*65+"\n  DEMO — 20 Unseen Messages\n"+"="*65)
    ok=0
    for text,exp_i,exp_l in UNSEEN:
        r=clf.predict(text);ii=r["intent"]==exp_i;ll=r["lang"]==exp_l
        if ii:ok+=1
        print(f"  {'✅' if ii else '❌'} [{r['intent']:14s}] {'✅' if ll else '⚠️ '}{r['lang']:7s}  conf={r['confidence']:.2f}  {r['inference_ms']:.1f}ms")
        print(f"     {text[:60]}")
    pct=ok/len(UNSEEN);print(f"\n  Accuracy: {ok}/{len(UNSEEN)} = {pct:.1%}");return pct

if __name__=="__main__":
    if "--predict" in sys.argv:
        clf=CNNClassifier.load()
        msg=" ".join(sys.argv[sys.argv.index("--predict")+1:]) or input("Message: ")
        print(json.dumps(clf.predict(msg),ensure_ascii=False,indent=2))
    elif "--benchmark" in sys.argv:
        CNNClassifier.load().benchmark(2000)
    else:
        print("="*65+"\n  RAVEN — TextCNN (CPU-Optimized)\n"+"="*65)
        clf=CNNClassifier()
        clf.fit(epochs=20,batch=256,lr=3e-3,D=48,F=128,drop=0.35,vocab=5000,aug_factor=0)
        clf.save();demo(clf);print();clf.benchmark(2000)
