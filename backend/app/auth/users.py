"""
User directory, loaded entirely from environment variables.

No usernames, passwords, emails or tokens are hardcoded anywhere in this
codebase. Until the real identifiers are provided, every account is a
placeholder read from `.env` (see `.env.example`) and the store is simply
empty — the app runs, `/api/health` is green, and `/api/auth/login` correctly
reports "no accounts configured" instead of crashing.

Each of the 5 accounts (ADMIN + USER_1..USER_4) supports two independent,
optional login mechanisms so the app works with whichever the operator
eventually provides:

  1. Username + password/secret  -> POST /api/auth/login
  2. A personal access link/token -> POST /api/auth/link-login

Env layout per account (all optional, all off until set):

    ADMIN_USER=<identifier>            USER_1=<identifier>
    ADMIN_PASSWORD=<secret>            USER_1_PASSWORD=<secret>
    ADMIN_ACCESS_TOKEN=<token>         USER_1_ACCESS_TOKEN=<token>
    ADMIN_NAME=<display name>          USER_1_NAME=<display name>
                                        USER_1_ROLE=user   (optional, defaults to "user")
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class User:
    identifier: str
    role: str
    display_name: str
    _password: str | None = field(default=None, repr=False)
    _access_token: str | None = field(default=None, repr=False)

    def check_password(self, candidate: str) -> bool:
        if not self._password or not candidate:
            return False
        return hmac.compare_digest(self._password, candidate)

    def check_access_token(self, candidate: str) -> bool:
        if not self._access_token or not candidate:
            return False
        return hmac.compare_digest(self._access_token, candidate)

    def to_public_dict(self) -> dict:
        return {"identifier": self.identifier, "role": self.role, "name": self.display_name}


class UserStore:
    """In-memory directory rebuilt from the environment at process start."""

    # (env var holding the identifier, prefix for the _PASSWORD/_ACCESS_TOKEN/
    # _ROLE/_NAME companions, default role, default display name). The admin
    # identifier lives in ADMIN_USER (not ADMIN) to match .env.example.
    _SLOTS = [
        ("ADMIN_USER", "ADMIN", "admin", "Administrateur"),
        ("USER_1", "USER_1", "user", "Utilisateur 1"),
        ("USER_2", "USER_2", "user", "Utilisateur 2"),
        ("USER_3", "USER_3", "user", "Utilisateur 3"),
        ("USER_4", "USER_4", "user", "Utilisateur 4"),
    ]

    def __init__(self, env: dict | None = None):
        env = env if env is not None else os.environ
        self._by_identifier: dict[str, User] = {}
        self._by_token: dict[str, User] = {}

        for identifier_var, prefix, default_role, default_name in self._SLOTS:
            identifier = env.get(identifier_var, "").strip()
            if not identifier:
                continue  # placeholder not yet provided — skip silently

            password = env.get(f"{prefix}_PASSWORD", "").strip() or None
            token = env.get(f"{prefix}_ACCESS_TOKEN", "").strip() or None
            role = env.get(f"{prefix}_ROLE", "").strip() or default_role
            name = env.get(f"{prefix}_NAME", "").strip() or default_name

            if prefix == "ADMIN":
                role = "admin"  # admin role is not operator-overridable

            user = User(
                identifier=identifier,
                role=role,
                display_name=name,
                _password=password,
                _access_token=token,
            )
            self._by_identifier[identifier] = user
            if token:
                self._by_token[token] = user

    def __len__(self) -> int:
        return len(self._by_identifier)

    def is_configured(self) -> bool:
        return len(self._by_identifier) > 0

    def find_by_identifier(self, identifier: str) -> User | None:
        return self._by_identifier.get((identifier or "").strip())

    def list_public(self) -> list[dict]:
        """Non-secret directory listing for the admin Settings page — never
        includes passwords or access tokens."""
        return [u.to_public_dict() for u in self._by_identifier.values()]

    def find_by_token(self, token: str) -> User | None:
        return self._by_token.get((token or "").strip())
