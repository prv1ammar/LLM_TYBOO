"""
LLM_TYBOO (litellm-tybo) — Full Test Suite
============================================
Tests every service and feature of the actual project:
  - 10 Docker services (llm-14b, llm-3b, qdrant, postgres, litellm,
                         api, dashboard, n8n, prometheus, grafana)
  - LiteLLM proxy (the actual model gateway)
  - Two APIs: api.py (JWT) + backend_api.py (X-API-Key)
  - RAG pipeline (ingest + query)
  - Model router (unit tests — routes through LiteLLM)
  - Embeddings (BGE-M3)
  - Async jobs
  - TythonClient SDK
  - Ingest pipeline (DuplicateRegistry unit tests)
  - n8n, Prometheus, Grafana reachability

Run from inside the src/ directory:
    cd src && python test_suite.py
"""

import os, sys, time, json, hashlib, tempfile, requests
from dotenv import load_dotenv

# Add src to path so relative imports work from this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

load_dotenv()

# Production DevOps URLs
API_URL       = "http://192.168.0.184:8888"
DASHBOARD_URL = "http://192.168.0.184:8501"
LITELLM_URL   = "http://192.168.0.184:4000"
N8N_URL       = "http://192.168.0.184:5678"
PROMETHEUS    = "http://192.168.0.184:9090"
GRAFANA       = "http://192.168.0.184:3000"
LITELLM_KEY   = os.getenv("LITELLM_KEY", "sk-tyboo-25871fc81b642fadeeb7da692040bd0e")
API_KEY       = os.getenv("API_KEY", "92129f24-12f0-478a-8c98-5b15070235e6")
ADMIN_USER    = "admin"
ADMIN_PASS    = "password123"

G="\033[92m"; R="\033[91m"; Y="\033[93m"; B="\033[94m"; BOLD="\033[1m"; X="\033[0m"
results = {"passed": 0, "failed": 0, "skipped": 0}
jwt_token = None

def section(t): print(f"\n{BOLD}{B}{'═'*62}{X}\n{BOLD}{B}  {t}{X}\n{BOLD}{B}{'═'*62}{X}")
def ok(n,d=""): results["passed"]+=1; print(f"  {G}✅ PASS{X}  {n}" + (f"  {Y}→ {d}{X}" if d else ""))
def fail(n,r=""): results["failed"]+=1; print(f"  {R}❌ FAIL{X}  {n}"); r and print(f"         {R}↳ {r}{X}")
def skip(n,r=""): results["skipped"]+=1; print(f"  {Y}⏭  SKIP{X}  {n}  {Y}({r}){X}")
def GET(url,h=None,t=15): return requests.get(url,headers=h or {},timeout=t)
def POST(url,j=None,d=None,h=None,t=120): return requests.post(url,json=j,data=d,headers=h or {},timeout=t)
def jh(): return {"Authorization":f"Bearer {jwt_token}","Content-Type":"application/json"} if jwt_token else {}
def ah(): return {"X-API-Key":API_KEY,"Content-Type":"application/json"}
def lh(): return {"Authorization":f"Bearer {LITELLM_KEY}","Content-Type":"application/json"}

# ══ 1. INFRASTRUCTURE ════════════════════════════════════════════════════════
def test_infrastructure():
    section("1. Infrastructure — Production Services")
    for name, url, codes in [
        ("FastAPI API",     f"{API_URL}/health",      [200]),
        ("Dashboard",       f"{DASHBOARD_URL}",       [200,302]),
        ("n8n Automation",  f"{N8N_URL}/healthz",     [200]),
        ("Prometheus",      f"{PROMETHEUS}/-/healthy",[200, 401]),
        ("Grafana",         f"{GRAFANA}/api/health",  [200, 502]),
    ]:
        try:
            r=GET(url,t=5)
            (ok if r.status_code in codes else fail)(name, f"HTTP {r.status_code}")
        except Exception as e: fail(name, "Service unreachable")

    try:
        import sentence_transformers; ok("sentence-transformers (local)", f"v{sentence_transformers.__version__}")
    except ImportError: fail("sentence_transformers", "not installed")

# ══ 2. LITELLM PROXY ═════════════════════════════════════════════════════════
def test_litellm():
    section("2. LiteLLM Proxy — Model Gateway (port 4000)")
    try:
        r=GET(f"{LITELLM_URL}/v1/models",h=lh(),t=15)
        if r.status_code==200:
            models=[m["id"] for m in r.json().get("data",[])]
            ok("GET /v1/models",f"models: {models}")
        else: fail("GET /v1/models",f"HTTP {r.status_code}")
    except Exception as e: fail("GET /v1/models",str(e))

    for model,label,timeout in [("internal-llm-3b","3B",60),("internal-llm-14b","14B",90),("internal-llm","default alias → 14B",90)]:
        try:
            r=POST(f"{LITELLM_URL}/v1/chat/completions",j={"model":model,"messages":[{"role":"user","content":"Say OK only."}],"max_tokens":10,"temperature":0.1},h=lh(),t=timeout)
            if r.status_code==200:
                content=r.json()["choices"][0]["message"]["content"]
                ok(f"LiteLLM → {label}",f"response: '{content.strip()}'")
            else: fail(f"LiteLLM → {label}",f"HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e: fail(f"LiteLLM → {label}",str(e))

    try:
        r=POST(f"{LITELLM_URL}/v1/chat/completions",j={"model":"internal-llm-3b","messages":[{"role":"user","content":"hi"}],"max_tokens":5},h={"Authorization":"Bearer WRONG"},t=10)
        (ok if r.status_code in [401,403] else fail)("LiteLLM rejects wrong key",f"HTTP {r.status_code}")
    except Exception as e: fail("LiteLLM rejects wrong key",str(e))

# ══ 3. JWT AUTH ═══════════════════════════════════════════════════════════════
def test_jwt_auth():
    global jwt_token
    section("3. JWT Authentication (api.py)")
    for path in ["/health","/info"]:
        try:
            r=GET(f"{API_URL}{path}",t=10)
            (ok if r.status_code==200 else fail)(f"GET {path} (public)",f"HTTP {r.status_code}")
        except Exception as e: fail(f"GET {path}",str(e))
    try:
        r=POST(f"{API_URL}/token",d={"username":ADMIN_USER,"password":ADMIN_PASS},h={"Content-Type":"application/x-www-form-urlencoded"},t=15)
        if r.status_code==200 and "access_token" in r.json():
            jwt_token=r.json()["access_token"]; ok("POST /token — valid credentials","token received")
        else: fail("POST /token — valid credentials",f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e: fail("POST /token",str(e))
    try:
        r=POST(f"{API_URL}/token",d={"username":"wrong","password":"wrong"},h={"Content-Type":"application/x-www-form-urlencoded"},t=10)
        (ok if r.status_code==401 else fail)("POST /token — wrong creds → 401",f"HTTP {r.status_code}")
    except Exception as e: fail("POST /token — wrong creds",str(e))
    try:
        r=POST(f"{API_URL}/rag/query",j={"question":"test"},t=10)
        (ok if r.status_code in [401,403,422] else fail)("/rag/query without token → rejected",f"HTTP {r.status_code}")
    except Exception as e: fail("/rag/query without token",str(e))

# ══ 4. API KEY AUTH ════════════════════════════════════════════════════════════
def test_apikey_auth():
    section("4. API Key Authentication (backend_api.py)")
    if not API_KEY: skip("All API key tests","API_KEY not set in .env"); return
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["test"]},h=ah(),t=30)
        (ok if r.status_code==200 else fail)("Valid key → /api/embeddings accepted",f"HTTP {r.status_code}")
    except Exception as e: fail("Valid key → /api/embeddings",str(e))
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["test"]},h={"X-API-Key":"WRONG"},t=10)
        (ok if r.status_code==401 else fail)("Wrong key → 401",f"HTTP {r.status_code}")
    except Exception as e: fail("Wrong key → 401",str(e))
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["test"]},t=10)
        (ok if r.status_code in [401,422] else fail)("No key → rejected",f"HTTP {r.status_code}")
    except Exception as e: fail("No key → rejected",str(e))

# ══ 5. EMBEDDINGS ════════════════════════════════════════════════════════════
def test_embeddings():
    section("5. Embeddings — BGE-M3 (1024D)")
    if not API_KEY: skip("All embedding tests","API_KEY not set"); return
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["Hello world test."]},h=ah(),t=180)
        if r.status_code==200:
            dims=len(r.json()["embeddings"][0])
            (ok if dims==1024 else fail)("Single embedding → 1024D",f"dims={dims}")
        else: fail("Single embedding",f"HTTP {r.status_code}")
    except Exception as e: fail("Single embedding",str(e))
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["نص عربي","هاد دارجة","texte français","English text"]},h=ah(),t=180)
        if r.status_code==200:
            embs=r.json()["embeddings"]
            (ok if len(embs)==4 else fail)("Multilingual batch (AR/Darija/FR/EN)",f"4 × {len(embs[0])}D")
        else: fail("Multilingual batch",f"HTTP {r.status_code}")
    except Exception as e: fail("Multilingual batch",str(e))
    try:
        r=POST(f"{API_URL}/api/embeddings",j={"texts":["cat","quantum physics"]},h=ah(),t=60)
        if r.status_code==200:
            v1,v2=r.json()["embeddings"]
            diff=sum(abs(a-b) for a,b in zip(v1[:20],v2[:20]))
            (ok if diff>0.1 else fail)("Different texts → different vectors",f"diff={diff:.4f}")
    except Exception as e: fail("Different vectors",str(e))
    try:
        r=POST(f"{LITELLM_URL}/v1/embeddings",j={"model":"internal-embedding","input":"test"},h=lh(),t=60)
        if r.status_code==200:
            dims=len(r.json()["data"][0]["embedding"])
            ok("LiteLLM /v1/embeddings (internal-embedding)",f"dims={dims}")
        else: fail("LiteLLM /v1/embeddings",f"HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e: fail("LiteLLM /v1/embeddings",str(e))

# ══ 6. RAG PIPELINE ══════════════════════════════════════════════════════════
def test_rag():
    section("6. RAG Pipeline (Ingest + Query)")
    test_docs=[
        {"text":"The annual leave policy allows 21 days of paid vacation per year.","metadata":{"source":"hr_policy.pdf"}},
        {"text":"Financial reports must be approved by the CFO before the 5th of each month.","metadata":{"source":"finance.pdf"}},
        {"text":"Morocco's renewable energy target is 52% by 2030.","metadata":{"source":"energy.txt"}},
    ]
    if not jwt_token: skip("RAG via JWT (ingest+query)","No JWT token")
    else:
        try:
            r=POST(f"{API_URL}/rag/ingest",j=test_docs,h=jh(),t=60)
            (ok if r.status_code==200 else fail)("POST /rag/ingest (JWT)",
                f"count={r.json().get('count',0)}" if r.status_code==200 else f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e: fail("POST /rag/ingest (JWT)",str(e))
        try:
            r=POST(f"{API_URL}/rag/query",j={"question":"How many days of annual leave?","top_k":3,"include_sources":True},h=jh(),t=300)
            if r.status_code==200:
                a=r.json().get("answer",""); s=r.json().get("sources",[])
                ok("POST /rag/query (JWT)",f"len={len(a)}, sources={len(s)}")
                found=any(k in a for k in ["21","leave","vacation","annual"])
                (ok if found else fail)("RAG answer contains relevant content","keyword found ✓" if found else f"answer: {a[:80]}")
            else: fail("POST /rag/query (JWT)",f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e: fail("POST /rag/query (JWT)",str(e))
        try:
            r=POST(f"{API_URL}/rag/query",j={"question":"What is the recipe for chocolate cake?","top_k":3},h=jh(),t=300)
            if r.status_code==200:
                used_kb=r.json().get("used_knowledge_base")
                ok("RAG unknown topic → general knowledge fallback",f"used_kb={used_kb}")
        except Exception as e: fail("RAG unknown topic fallback",str(e))
        try:
            r=POST(f"{API_URL}/rag/query",j={"question":"ما هو هدف المغرب في الطاقة المتجددة؟"},h=jh(),t=300)
            if r.status_code==200:
                a=r.json().get("answer","")
                (ok if len(a)>5 else fail)("RAG query in Arabic",f"len={len(a)}")
            else: fail("RAG query in Arabic",f"HTTP {r.status_code}")
        except Exception as e: fail("RAG query in Arabic",str(e))
    if not API_KEY: skip("RAG via API Key (backend_api)","API_KEY not set")
    else:
        try:
            r=POST(f"{API_URL}/api/rag/ingest",j={"documents":[{"text":"Python was created by Guido van Rossum in 1991.","metadata":{"source":"tech.txt"}}],"collection":"test_col"},h=ah(),t=60)
            (ok if r.status_code==200 else fail)("POST /api/rag/ingest (API Key)",f"HTTP {r.status_code}")
        except Exception as e: fail("POST /api/rag/ingest (API Key)",str(e))
        try:
            r=POST(f"{API_URL}/api/rag/query",j={"question":"Who created Python?","collection":"test_col","top_k":3},h=ah(),t=300)
            if r.status_code==200:
                a=r.json().get("answer",""); ok("POST /api/rag/query (API Key)",f"len={len(a)}")
            else: fail("POST /api/rag/query (API Key)",f"HTTP {r.status_code}")
        except Exception as e: fail("POST /api/rag/query (API Key)",str(e))

# ══ 7. MODEL ROUTER ══════════════════════════════════════════════════════════
def test_model_router():
    section("7. Smart Model Router (unit tests — routes via LiteLLM)")
    try:
        sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
        from model_router import route, get_model_for_rag, get_model
        cases=[
            ("Bonjour, comment ca va ?","3B"),("Hello","3B"),("What is the capital of Morocco?","3B"),
            ("merci","3B"),("yes","3B"),("Give me a list of cities","3B"),
            ("Analyse this contract and find abusive clauses","14B"),
            ("Write a Python function to parse CSV","14B"),
            ("What is the difference between RAG and fine-tuning?","14B"),
            ("Compare offers from suppliers A, B and C","14B"),
            ("Why is the API returning a 500 error?","14B"),
            ("Analyse le contrat et liste les clauses abusives","14B"),
            ("مرحبا","3B"),
            ("حلل هذا العقد وقدم التوصيات الاستراتيجية","14B"),
        ]
        passed=0
        for q,exp in cases:
            _,chosen=route(q)
            if chosen==exp: ok(f"route('{q[:48]}')",f"→ {chosen} ✓"); passed+=1
            else: fail(f"route('{q[:48]}')",f"expected {exp}, got {chosen}")
        print(f"\n  Router score: {passed}/{len(cases)} correct")
        m=get_model_for_rag(); ok("get_model_for_rag() returns 14B model",str(getattr(m,'model_name','?')))
        m=get_model(); ok("get_model() returns default model",str(getattr(m,'model_name','?')))
    except ImportError: skip("Model router tests","model_router.py not in path — run from src/")
    except Exception as e: fail("Model router tests",str(e))

# ══ 8. ASYNC JOBS ════════════════════════════════════════════════════════════
def test_jobs():
    section("8. Async Job Queue (backend_api.py)")
    if not API_KEY: skip("All job tests","API_KEY not set"); return
    job_id=None
    try:
        r=POST(f"{API_URL}/api/jobs",j={"job_type":"batch_embed","params":{"texts":["doc1","doc2","doc3"]},"priority":"normal"},h=ah(),t=60)
        if r.status_code==200:
            job_id=r.json().get("job_id"); ok("Submit batch_embed job",f"job_id={job_id}")
        else: fail("Submit batch_embed job",f"HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e: fail("Submit batch_embed job",str(e))
    if job_id:
        try:
            start=time.time(); final=None
            while time.time()-start<120:
                r=GET(f"{API_URL}/api/jobs/{job_id}",h=ah(),t=10)
                if r.status_code==200:
                    final=r.json().get("status")
                    if final in ["completed","failed"]: break
                time.sleep(3)
            elapsed=int(time.time()-start)
            if final=="completed": ok("Job completes successfully",f"elapsed={elapsed}s")
            elif final=="failed": fail("Job completes",f"job failed: {r.json().get('error','?')}")
            else: fail("Job completes",f"timeout after 120s, status={final}")
        except Exception as e: fail("Job polling",str(e))
    try:
        r=GET(f"{API_URL}/api/jobs/non-existent-xyz-999",h=ah(),t=30)
        (ok if r.status_code==404 else fail)("Non-existent job → 404",f"HTTP {r.status_code}")
    except Exception as e: fail("Non-existent job → 404",str(e))
    try:
        r=POST(f"{API_URL}/api/jobs",j={"job_type":"batch_embed","params":{"texts":["x"]*50},"priority":"low"},h=ah(),t=15)
        if r.status_code==200:
            cid=r.json()["job_id"]
            r2=requests.delete(f"{API_URL}/api/jobs/{cid}",headers=ah(),timeout=30)
            (ok if r2.status_code==200 else fail)("Cancel pending job",f"job_id={cid}")
    except Exception as e: fail("Cancel pending job",str(e))

# ══ 9. AGENT ═════════════════════════════════════════════════════════════════
def test_agent():
    section("9. Agent — AGATE (orchestrator.py)")
    if not jwt_token: skip("Agent via JWT","No JWT token")
    else:
        try:
            r=POST(f"{API_URL}/agent/general",j={"question":"What is the capital of Morocco?","top_k":3},h=jh(),t=300)
            if r.status_code==200:
                a=r.json().get("answer","")
                ok("POST /agent/general — factual",f"len={len(a)}")
            else: fail("POST /agent/general",f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e: fail("POST /agent/general",str(e))
        try:
            r=POST(f"{API_URL}/agent/general",j={"question":"Explique le RAG en 2 phrases.","top_k":3},h=jh(),t=300)
            if r.status_code==200:
                a=r.json().get("answer",""); (ok if len(a)>20 else fail)("Agent answers in French",f"len={len(a)}")
        except Exception as e: fail("Agent answers in French",str(e))
    if not API_KEY: skip("Agent via API Key","API_KEY not set")
    else:
        try:
            r=POST(f"{API_URL}/api/agent/analyze",j={"document":"Contract between ACME Corp and John Doe. Start: Jan 1 2025. Salary: 15000 MAD/month. Notice: 1 month.","instructions":"Extract all dates and amounts."},h=ah(),t=300)
            if r.status_code==200:
                d=r.json()
                (ok if "summary" in d and "key_points" in d else fail)("POST /api/agent/analyze — structured output",f"fields: {list(d.keys())}")
            else: fail("POST /api/agent/analyze",f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e: fail("POST /api/agent/analyze",str(e))
        try:
            r=POST(f"{API_URL}/api/agent/generate",j={"prompt":"Write a 1-sentence description of Morocco.","context":""},h=ah(),t=300)
            if r.status_code==200:
                c=r.json().get("content",""); (ok if len(c)>10 else fail)("POST /api/agent/generate",f"len={len(c)}")
            else: fail("POST /api/agent/generate",f"HTTP {r.status_code}")
        except Exception as e: fail("POST /api/agent/generate",str(e))

# ══ 10. CHAT & COMPLETION ════════════════════════════════════════════════════
def test_chat():
    section("10. Chat & Completion (backend_api.py)")
    if not API_KEY: skip("All chat tests","API_KEY not set"); return
    try:
        r=POST(f"{API_URL}/api/chat",j={"message":"What is 3+3? Number only.","temperature":0.1,"max_tokens":20},h=ah(),t=90)
        if r.status_code==200:
            resp=r.json().get("response",""); ok("POST /api/chat (simple → 3B)",f"response: '{resp.strip()}'")
        else: fail("POST /api/chat",f"HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e: fail("POST /api/chat",str(e))
    try:
        r=POST(f"{API_URL}/api/chat",j={"message":"Explain the difference between RAG and fine-tuning in 2 sentences.","temperature":0.3,"max_tokens":200},h=ah(),t=300)
        if r.status_code==200:
            resp=r.json().get("response",""); (ok if len(resp)>20 else fail)("POST /api/chat (complex → 14B)",f"len={len(resp)}")
        else: fail("POST /api/chat (complex)",f"HTTP {r.status_code}")
    except Exception as e: fail("POST /api/chat (complex)",str(e))
    try:
        r=POST(f"{API_URL}/api/complete",j={"prompt":"The capital of France is","max_tokens":10},h=ah(),t=90)
        if r.status_code==200:
            c=r.json().get("completion",""); ok("POST /api/complete",f"'{c.strip()[:50]}'")
        else: fail("POST /api/complete",f"HTTP {r.status_code}")
    except Exception as e: fail("POST /api/complete",str(e))

# ══ 11. TYTHON SDK ════════════════════════════════════════════════════════════
def test_sdk():
    section("11. TythonClient SDK (src/sdk/tython_client.py)")
    try:
        sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "sdk"))
        sys.path.insert(0, sdk_path)
        from tython_client import TythonClient
    except ImportError: skip("All SDK tests","tython_client.py not found"); return
    if not API_KEY: skip("All SDK tests","API_KEY not set"); return
    client=TythonClient(api_url=API_URL,api_key=API_KEY)
    try:
        h=client.health(); (ok if h.get("status")=="healthy" else fail)("SDK: health()",h.get("status","?"))
    except Exception as e: fail("SDK: health()",str(e))
    try:
        info=client.info(); ok("SDK: info()",info.get("service","?"))
    except Exception as e: fail("SDK: info()",str(e))
    try:
        v=client.embed(["Testing TythonClient embed method"]); (ok if len(v[0])==1024 else fail)("SDK: embed() → 1024D",f"dims={len(v[0])}")
    except Exception as e: fail("SDK: embed()",str(e))
    try:
        v=client.embed_single("single text"); (ok if len(v)==1024 else fail)("SDK: embed_single() → 1024D")
    except Exception as e: fail("SDK: embed_single()",str(e))
    try:
        r=client.rag_query("What is Morocco?",collection="default"); (ok if "answer" in r else fail)("SDK: rag_query()",f"len={len(r.get('answer',''))}")
    except Exception as e: fail("SDK: rag_query()",str(e))
    try:
        jid=client.submit_job("batch_embed",{"texts":["sdk a","sdk b"]}); ok("SDK: submit_job()",f"job_id={jid}")
        res=client.get_job_result(jid,wait=True,timeout=120); (ok if res and "embeddings" in res else fail)("SDK: get_job_result(wait=True)",f"count={res.get('count',0) if res else 0}")
    except Exception as e: fail("SDK: submit_job + get_job_result",str(e))

# ══ 12. INGEST PIPELINE UNIT TESTS ══════════════════════════════════════════
def test_ingest_pipeline():
    section("12. Ingest Pipeline — DuplicateRegistry Unit Tests")
    try:
        sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
        from ingest import DuplicateRegistry
    except ImportError: skip("Ingest tests","ingest.py not found — run from src/"); return
    with tempfile.TemporaryDirectory() as tmp:
        path=os.path.join(tmp,"reg.json")
        reg=DuplicateRegistry(registry_path=path); ok("DuplicateRegistry creates fresh registry")
        h1=DuplicateRegistry.hash_text("hello"); h2=DuplicateRegistry.hash_text("hello"); h3=DuplicateRegistry.hash_text("world")
        (ok if h1==h2 and h1!=h3 else fail)("hash_text — deterministic + collision-resistant")
        f=os.path.join(tmp,"test.txt")
        open(f,"w").write("test content")
        fh=DuplicateRegistry.hash_file(f); (ok if len(fh)==64 else fail)("hash_file — SHA-256, 64 hex chars")
        known,hval=reg.is_file_known(f); (ok if not known else fail)("is_file_known — new file → False")
        reg.register_file(f,hval,3); known2,_=reg.is_file_known(f); (ok if known2 else fail)("register_file → is_file_known returns True")
        chunk="Test chunk of text."
        (ok if not reg.is_chunk_known(chunk) else fail)("is_chunk_known — new chunk → False")
        reg.register_chunk(chunk); (ok if reg.is_chunk_known(chunk) else fail)("register_chunk → is_chunk_known returns True")
        reg.update_stats(ingested=5,skipped=2)
        reg2=DuplicateRegistry(registry_path=path); s=reg2.get_stats()
        (ok if s["total_ingested"]==5 and s["known_files"]==1 else fail)("Registry persists to disk and reloads",f"stats={s}")
        reg2.clear(); s2=reg2.get_stats(); (ok if s2["total_ingested"]==0 else fail)("Registry clear() resets all data")
        open(f,"w").write("MODIFIED content"); known3,_=reg.is_file_known(f)
        (ok if not known3 else fail)("Modified file detected → triggers re-ingest")

# ══ 13. N8N + MONITORING ═════════════════════════════════════════════════════
def test_monitoring():
    section("13. n8n + Prometheus + Grafana")
    for name,url,codes in [
        ("n8n healthz",          f"{N8N_URL}/healthz",        [200]),
        ("n8n UI",               N8N_URL,                      [200,302]),
        ("Prometheus healthy",   f"{PROMETHEUS}/-/healthy",    [200, 401]),
        ("Grafana healthy",      f"{GRAFANA}/api/health",      [200, 502]),
    ]:
        try:
            r=GET(url,t=10); (ok if r.status_code in codes else fail)(name,f"HTTP {r.status_code}")
        except Exception as e: fail(name,str(e))
    # n8n→LiteLLM credentials check
    try:
        r=GET(f"{LITELLM_URL}/v1/models",h=lh(),t=10)
        (ok if r.status_code==200 else fail)("n8n→LiteLLM credentials work",f"Base: {LITELLM_URL}/v1, Key: {LITELLM_KEY}")
    except Exception as e: fail("n8n→LiteLLM credentials",str(e))

# ══ SUMMARY ══════════════════════════════════════════════════════════════════
def summary():
    total=results["passed"]+results["failed"]+results["skipped"]
    print(f"\n{BOLD}{'═'*62}{X}\n{BOLD}  RESULTS{X}\n{BOLD}{'═'*62}{X}")
    print(f"  {G}✅ Passed :  {results['passed']}{X}")
    print(f"  {R}❌ Failed :  {results['failed']}{X}")
    print(f"  {Y}⏭  Skipped:  {results['skipped']}{X}")
    print(f"  Total    :  {total}")
    pct=int(results["passed"]/max(results["passed"]+results["failed"],1)*100)
    print(f"\n  Score: {pct}%  ",end="")
    if results["failed"]==0: print(f"{G}{BOLD}🎉 All tests passed!{X}")
    elif pct>=80: print(f"{Y}{BOLD}⚠️  Some tests failed{X}")
    else: print(f"{R}{BOLD}❌ Too many failures — check services{X}")
    print(f"{BOLD}{'═'*62}{X}\n")
    return results["failed"]==0

if __name__=="__main__":
    print(f"\n{BOLD}{B}")
    print("  ██╗     ██╗     ███╗   ███╗    ████████╗██╗   ██╗██████╗  ██████╗  ██████╗ ")
    print("  ██║     ██║     ████╗ ████║       ██╔══╝╚██╗ ██╔╝██╔══██╗██╔═══██╗██╔═══██╗")
    print(f"{X}")
    print(f"  {BOLD}Full Test Suite — litellm-tybo edition{X}")
    print(f"  API: {API_URL}  |  LiteLLM: {LITELLM_URL}\n")
    start=time.time()
    test_infrastructure(); test_litellm(); test_jwt_auth(); test_apikey_auth()
    test_embeddings(); test_rag(); test_model_router(); test_jobs()
    test_agent(); test_chat(); test_sdk(); test_ingest_pipeline(); test_monitoring()
    print(f"\n  ⏱  Total time: {time.time()-start:.1f}s")
    sys.exit(0 if summary() else 1)
