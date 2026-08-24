"""Login endpoints. Two independent mechanisms, both optional — whichever
the operator populates in `.env` works; see `app/auth/users.py`."""
from __future__ import annotations

from flask import Blueprint, current_app, g, jsonify, request

from app import extensions
from app.auth.decorators import login_required
from app.auth.jwt_utils import issue_token

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _token_response(user):
    cfg = current_app.config
    token, expires_at = issue_token(user, cfg["SECRET_KEY"], cfg["JWT_ALGORITHM"], cfg["JWT_EXPIRES_MINUTES"])
    return jsonify(token=token, expires_at=expires_at.isoformat(), user=user.to_public_dict())


@bp.post("/login")
def login():
    store = extensions.user_store
    if store is None or not store.is_configured():
        return jsonify(
            error="no_accounts_configured",
            message="No user accounts are configured yet. Set ADMIN_USER/ADMIN_PASSWORD "
                    "(and USER_1..USER_4) in the backend environment — see .env.example.",
        ), 503

    body = request.get_json(silent=True) or {}
    identifier = str(body.get("identifier", "")).strip()
    password = str(body.get("password", ""))
    if not identifier or not password:
        return jsonify(error="bad_request", message="identifier and password are required."), 400

    user = store.find_by_identifier(identifier)
    if user is None or not user.check_password(password):
        return jsonify(error="invalid_credentials", message="Incorrect identifier or password."), 401

    return _token_response(user)


@bp.post("/link-login")
def link_login():
    """Alternative login for a personal access link, e.g. `/access/<token>`
    on the frontend, in case the 5 accounts are distributed as links rather
    than username/password pairs."""
    store = extensions.user_store
    if store is None or not store.is_configured():
        return jsonify(
            error="no_accounts_configured",
            message="No user accounts are configured yet.",
        ), 503

    body = request.get_json(silent=True) or {}
    token = str(body.get("token", "")).strip()
    if not token:
        return jsonify(error="bad_request", message="token is required."), 400

    user = store.find_by_token(token)
    if user is None:
        return jsonify(error="invalid_token", message="This access link is not recognized."), 401

    return _token_response(user)


@bp.get("/me")
@login_required
def me():
    cfg = current_app.config
    role = g.current_user["role"]
    scopes = ["*"] if role == "admin" else cfg["ROLE_PERMISSIONS"].get(role, [])
    return jsonify(user=g.current_user, scopes=scopes)


@bp.post("/logout")
@login_required
def logout():
    # Stateless JWTs — nothing to invalidate server-side. Endpoint exists so
    # the frontend has a single consistent call to make on sign-out.
    return jsonify(message="Logged out.")
