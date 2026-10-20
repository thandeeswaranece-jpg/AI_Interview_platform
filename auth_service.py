# auth_service.py

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt


# ============================================================
# JWT CONFIGURATION
# ============================================================

SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "ai-interview-platform-secret-key-change-this"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    """

    password_bytes = password.encode("utf-8")

    # bcrypt supports a maximum of 72 bytes
    if len(password_bytes) > 72:
        raise ValueError(
            "Password must be 72 bytes or fewer."
        )

    hashed_password = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed_password.decode("utf-8")


# ============================================================
# PASSWORD VERIFICATION
# ============================================================

def verify_password(
    password: str,
    password_hash: str
) -> bool:
    """
    Verify a password against the stored bcrypt hash.
    """

    password_bytes = password.encode("utf-8")
    hash_bytes = password_hash.encode("utf-8")

    # bcrypt maximum input size
    if len(password_bytes) > 72:
        return False

    try:
        return bcrypt.checkpw(
            password_bytes,
            hash_bytes
        )

    except (ValueError, TypeError):
        return False


# ============================================================
# CREATE JWT ACCESS TOKEN
# ============================================================

def create_access_token(
    user_id: int,
    email: str,
    expires_delta: timedelta | None = None
) -> str:
    """
    Create a JWT token containing the user's ID and email.
    """

    if expires_delta is None:
        expires_delta = timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    expire = (
        datetime.now(timezone.utc)
        + expires_delta
    )

    payload = {
        "user_id": user_id,
        "email": email,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


# ============================================================
# DECODE / VERIFY JWT ACCESS TOKEN
# ============================================================

def decode_access_token(token: str):
    """
    Decode and verify a JWT token.

    Returns:
        dict  -> valid token payload
        None  -> invalid/expired token
    """

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # Make sure required user information exists
        if "user_id" not in payload:
            return None

        if "email" not in payload:
            return None

        return payload

    except JWTError:
        return None

    except Exception:
        return None