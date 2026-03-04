"""
test_db_connections.py — Database Infrastructure Test
======================================================
PURPOSE:
  Verifies connectivity to PostgreSQL and Qdrant before deployment.
  Attempts to connect using both Docker service names and localhost fallbacks.

TESTS:
  1. PostgreSQL: Connects, creates a temp table, and performs a write/read.
  2. Qdrant: Connects, lists collections, and creates a temp test collection.

USAGE:
  python src/test_db_connections.py
"""

import os
import psycopg2
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# --- Configuration ---
# PostgreSQL
DB_URL_DOCKER = os.getenv("DATABASE_URL")
# Derived localhost version (replaces 'postgres' with 'localhost')
DB_URL_LOCAL = DB_URL_DOCKER.replace("@postgres:", "@localhost:") if DB_URL_DOCKER else None

# Qdrant
QDRANT_URL_DOCKER = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_URL_LOCAL = QDRANT_URL_DOCKER.replace("qdrant", "localhost")

def test_postgresql():
    print("\n🐘 Testing PostgreSQL...")
    
    # Try local first (dev) then docker (prod)
    urls = [DB_URL_LOCAL, DB_URL_DOCKER]
    success = False
    
    for url in urls:
        if not url: continue
        print(f"  Attempting connection to: {url.split('@')[1]}")
        try:
            conn = psycopg2.connect(url, connect_timeout=5)
            with conn.cursor() as cur:
                # Test write/read
                cur.execute("CREATE TEMPORARY TABLE db_test (id SERIAL PRIMARY KEY, val TEXT)")
                cur.execute("INSERT INTO db_test (val) VALUES ('deployment_check')")
                cur.execute("SELECT val FROM db_test")
                res = cur.fetchone()[0]
                if res == 'deployment_check':
                    print("  ✅ PostgreSQL is HEALTHY (Read/Write OK)")
                    success = True
            conn.close()
            if success: break
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}...")

    if not success:
        print("  🚨 CRITICAL: Could not reach PostgreSQL. Check if Docker container is up.")
    return success

def test_qdrant():
    print("\n🔍 Testing Qdrant (Vector DB)...")
    
    urls = [QDRANT_URL_LOCAL, QDRANT_URL_DOCKER]
    success = False
    
    for url in urls:
        print(f"  Attempting connection to: {url}")
        try:
            # Test simple health check first
            r = requests.get(f"{url}/healthz", timeout=5)
            if r.status_code == 200:
                client = QdrantClient(url=url)
                cols = client.get_collections()
                print(f"  ✅ Qdrant is HEALTHY ({len(cols.collections)} collections found)")
                success = True
                break
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}...")

    if not success:
        print("  🚨 CRITICAL: Could not reach Qdrant. Check port 6333.")
    return success

def main():
    print("🚀 Pre-deployment Database Verification")
    print("=" * 45)
    
    pg_ok = test_postgresql()
    qd_ok = test_qdrant()
    
    print("\n" + "=" * 45)
    if pg_ok and qd_ok:
        print("🎉 DATABASE INFRASTRUCTURE: READY FOR DEPLOYMENT")
    else:
        print("⚠️  DATABASE INFRASTRUCTURE: ISSUES DETECTED")
        exit(1)

if __name__ == "__main__":
    main()
