"""
users.py — PostgreSQL User Management
=======================================
PURPOSE:
  Replaces the hardcoded USERS_DB dict in api.py with a real PostgreSQL table.
  Users are stored in the 'tyboo_users' table — no code changes needed to add/remove users.

TABLE SCHEMA:
  tyboo_users (
    username        TEXT PRIMARY KEY,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',   -- 'admin' or 'user'
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
  )

ROLES:
  admin  → can call /admin/users/* endpoints (create, list, delete users)
  user   → can use RAG, agent, chat — cannot manage other users

AUTO-SEED:
  On first startup, if the table is empty, creates the admin user from env vars:
    ADMIN_USERNAME  (default: admin)
    ADMIN_PASSWORD  (default: password123 — CHANGE THIS)

HOW TO ADD USERS WITHOUT TOUCHING CODE:
  Option 1 — API endpoint (recommended):
    POST /admin/users  with JWT token of an admin user

  Option 2 — psql directly:
    INSERT INTO tyboo_users (username, hashed_password, role)
    VALUES ('newuser', '<bcrypt_hash>', 'user');

  Option 3 — Streamlit dashboard:
    Tab "User Management" → Add User form
"""

import os
import psycopg2
import psycopg2.extras
from typing import Optional, List, Dict
from auth import get_password_hash
import time
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://litellm:litellm_password@postgres:5432/litellm_db")


def get_conn():
    """Open a new PostgreSQL connection. Close it after each operation."""
    return psycopg2.connect(DATABASE_URL)


def init_users_table():
    """
    Create the tyboo_users table if it doesn't exist yet.
    Retries multiple times to wait for Postgres to become ready.
    """
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    # Create table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tyboo_users (
                            username        TEXT PRIMARY KEY,
                            hashed_password TEXT NOT NULL,
                            role            TEXT NOT NULL DEFAULT 'user',
                            is_active       BOOLEAN NOT NULL DEFAULT TRUE,
                            created_at      TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    conn.commit()

                    # Seed admin user if table is empty
                    cur.execute("SELECT COUNT(*) FROM tyboo_users")
                    count = cur.fetchone()[0]
                    if count == 0:
                        admin_username = os.getenv("ADMIN_USERNAME", "admin")
                        admin_password = os.getenv("ADMIN_PASSWORD", "password123")
                        cur.execute(
                            """INSERT INTO tyboo_users (username, hashed_password, role)
                               VALUES (%s, %s, 'admin')
                               ON CONFLICT (username) DO NOTHING""",
                            (admin_username, get_password_hash(admin_password))
                        )
                        conn.commit()
                        print(f"[Users] Seeded default admin user: '{admin_username}'", flush=True)
                    else:
                        print(f"[Users] tyboo_users table ready — {count} user(s) found", flush=True)
            finally:
                conn.close()
            return  # Success, exit retry loop
        except psycopg2.OperationalError as e:
            print(f"[Users] DB not ready on attempt {attempt}/{max_retries}. Waiting...", flush=True)
            if attempt == max_retries:
                print(f"[WARN] Failed to initialize users table after {max_retries} attempts: {e}", flush=True)
                raise
            time.sleep(2)
        except Exception as e:
            print(f"[WARN] Could not initialize users table: {e}", flush=True)
            break


def get_user(username: str) -> Optional[Dict]:
    """
    Fetch a single user by username.

    Returns:
        Dict with username, hashed_password, role, is_active
        None if user doesn't exist or is inactive
    """
    try:
        conn = get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT username, hashed_password, role, is_active FROM tyboo_users WHERE username = %s",
                    (username,)
                )
                row = cur.fetchone()
                if row and row["is_active"]:
                    return dict(row)
                return None
        finally:
            conn.close()
    except Exception as e:
        print(f"[ERROR] get_user failed: {str(e)}", flush=True)
        raise


def list_users() -> List[Dict]:
    """
    Return all users (without hashed_password for security).

    Used by the admin dashboard to display the user list.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT username, role, is_active, created_at FROM tyboo_users ORDER BY created_at DESC"
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def create_user(username: str, password: str, role: str = "user") -> Dict:
    """
    Create a new user with a bcrypt-hashed password.

    Args:
        username: Must be unique — raises ValueError if already exists
        password: Plain text — hashed before storage
        role:     'admin' or 'user'

    Returns:
        Dict with the new user's info (no password)

    Raises:
        ValueError: If username already exists
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """INSERT INTO tyboo_users (username, hashed_password, role)
                       VALUES (%s, %s, %s)""",
                    (username, get_password_hash(password), role)
                )
                conn.commit()
                return {"username": username, "role": role, "is_active": True}
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise ValueError(f"Username '{username}' already exists")
    finally:
        conn.close()


def delete_user(username: str) -> bool:
    """
    Permanently delete a user from the database.

    Returns:
        True if deleted, False if user not found
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tyboo_users WHERE username = %s", (username,))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        conn.close()


def update_password(username: str, new_password: str) -> bool:
    """
    Update a user's password.

    Returns:
        True if updated, False if user not found
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tyboo_users SET hashed_password = %s WHERE username = %s",
                (get_password_hash(new_password), username)
            )
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def set_user_active(username: str, is_active: bool) -> bool:
    """Enable or disable a user account without deleting it."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tyboo_users SET is_active = %s WHERE username = %s",
                (is_active, username)
            )
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()
