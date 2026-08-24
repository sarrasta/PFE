"""Minimal JWT issue/verify helpers built on PyJWT."""
from __future__ import annotations

import datetime as dt

import jwt

from app.auth.users import User


def issue_token(user: User, secret: str, algorithm: str, expires_minutes: int) -> tuple[str, dt.datetime]:
    now = dt.datetime.now(dt.timezone.utc)
    expires_at = now + dt.timedelta(minutes=expires_minutes)
    payload = {
        "sub": user.identifier,
        "role": user.role,
        "name": user.display_name,
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token, expires_at


def decode_token(token: str, secret: str, algorithm: str) -> dict:
    """Raises jwt.PyJWTError subclasses on invalid/expired tokens."""
    return jwt.decode(token, secret, algorithms=[algorithm])
