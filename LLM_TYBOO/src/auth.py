"""
auth.py — JWT Authentication
==============================
PURPOSE:
  Handles user authentication for the FastAPI endpoints.
  Users log in with username/password and receive a JWT token.
  That token is then included in the Authorization header of subsequent requests.

HOW JWT WORKS IN THIS PROJECT:
  1. POST /token with {username, password}
  2. Server verifies credentials and returns {"access_token": "...", "token_type": "bearer"}
  3. Client includes the token in every protected request:
     Authorization: Bearer <token>
  4. get_current_user() decodes the token and returns the user dict
  5. FastAPI route dependencies call get_current_user() to enforce auth

TOKEN EXPIRY:
  Tokens expire after ACCESS_TOKEN_EXPIRE_MINUTES (default: 60 minutes).
  After expiry, the user must log in again to get a new token.

SECURITY NOTES:
  - JWT_SECRET_KEY MUST be changed in production — generate with: openssl rand -hex 32
  - Passwords are hashed with bcrypt — never stored in plain text
  - The default "admin" / "password123" credentials MUST be changed before deployment

HOW TO USE:
  # In a FastAPI route:
  from auth import get_current_user

  @app.get("/protected")
  async def protected_route(current_user: dict = Depends(get_current_user)):
      return {"message": f"Hello {current_user['username']}"}

  # To hash a password (when creating new users):
  from auth import get_password_hash
  hashed = get_password_hash("my_secure_password")

  # To verify a password:
  from auth import verify_password
  is_valid = verify_password("plain_text", hashed_password)
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv

load_dotenv()

# JWT configuration
# SECRET_KEY signs and verifies tokens — anyone with this key can forge tokens
# NEVER commit the real value to git — set it via .env
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-generate-with-openssl-rand-hex-32")
ALGORITHM = "HS256"  # HMAC-SHA256 — standard and secure for JWTs
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Tokens expire after 1 hour

# Password hashing context
# bcrypt is the industry standard for password hashing — slow by design to resist brute force
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme — tells FastAPI where to find the token (Authorization: Bearer header)
# tokenUrl points to the /token endpoint that issues tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain text password matches a bcrypt hash.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        plain_password:  The password the user typed
        hashed_password: The bcrypt hash stored in the database

    Returns:
        True if they match, False otherwise

    Usage:
        if not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a plain text password using bcrypt.

    Call this when creating a new user or changing a password.
    The resulting hash is safe to store in a database.

    Args:
        password: Plain text password to hash

    Returns:
        bcrypt hash string (starts with "$2b$...")

    Usage:
        hashed = get_password_hash("my_secure_password")
        # Store hashed in your user database, never store the plain text
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT token containing the given data.

    The token encodes the user identity ({"sub": username}) plus an expiry time.
    It is signed with SECRET_KEY so the server can verify it was issued by us.

    Args:
        data:          Dict to encode in the token (typically {"sub": username})
        expires_delta: How long until the token expires. Defaults to 15 minutes
                       if not specified. Pass ACCESS_TOKEN_EXPIRE_MINUTES for login.

    Returns:
        JWT token string — send this to the client

    Usage:
        token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency — decode and validate a JWT token from the request header.

    This function is used as a dependency in protected routes:
      @app.post("/rag/query", dependencies=[Depends(get_current_user)])

    FastAPI automatically extracts the token from the Authorization: Bearer header,
    passes it here, and raises HTTPException if the token is missing or invalid.

    Args:
        token: JWT string extracted from the Authorization header by FastAPI

    Returns:
        Dict with user information: {"username": "..."}
        In production, you would fetch the full user record from the database here.

    Raises:
        HTTPException(401): If the token is missing, expired, or has an invalid signature
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials — token may be missing or expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # In production: fetch full user record from PostgreSQL here
    # For now, return a minimal dict with just the username
    return {"username": username}
