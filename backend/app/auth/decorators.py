"""Route protection decorators — every non-public route in the app goes
through `login_required`, and anything role-sensitive additionally goes
through `scope_required`. The frontend never decides access on its own;
it only reflects what these decorators already enforce server-side.
"""
from __future__ import annotations

import functools

import jwt
from flask import current_app, g, jsonify, request

from app.auth.jwt_utils import decode_token


def _extract_bearer_token() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):].strip()
    return None


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        token = _extract_bearer_token()
        if not token:
            return jsonify(error="unauthorized", message="Missing bearer token."), 401
        cfg = current_app.config
        try:
            payload = decode_token(token, cfg["SECRET_KEY"], cfg["JWT_ALGORITHM"])
        except jwt.ExpiredSignatureError:
            return jsonify(error="token_expired", message="Session expired, please log in again."), 401
        except jwt.PyJWTError:
            return jsonify(error="invalid_token", message="Invalid session token."), 401

        g.current_user = {
            "identifier": payload.get("sub"),
            "role": payload.get("role"),
            "name": payload.get("name"),
        }
        return view(*args, **kwargs)

    return wrapped


def _role_has_scope(role: str, scope: str, role_permissions: dict) -> bool:
    if role == "admin":
        return True
    return scope in role_permissions.get(role, [])


def scope_required(scope: str):
    """Require the caller's role to include `scope`. Must be stacked under
    `login_required`. Admin always passes."""

    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            user = getattr(g, "current_user", None)
            if user is None:
                return jsonify(error="unauthorized", message="Missing bearer token."), 401
            role_permissions = current_app.config["ROLE_PERMISSIONS"]
            if not _role_has_scope(user["role"], scope, role_permissions):
                return jsonify(
                    error="forbidden",
                    message=f"Your role does not have access to '{scope}'.",
                ), 403
            return view(*args, **kwargs)

        return wrapped

    return decorator


def admin_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        user = getattr(g, "current_user", None)
        if user is None:
            return jsonify(error="unauthorized", message="Missing bearer token."), 401
        if user["role"] != "admin":
            return jsonify(error="forbidden", message="Administrator access required."), 403
        return view(*args, **kwargs)

    return wrapped
