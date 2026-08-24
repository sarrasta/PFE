"""
Central configuration, loaded entirely from environment variables.

Nothing sensitive is hardcoded here. See `.env.example` at the repo root for
every variable this file reads, and README.md ("Environment variables" /
"How to add the 4 users and admin later") for how to populate them.
"""
import os


def _bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str = "") -> list[str]:
    val = os.environ.get(name, default)
    return [v.strip() for v in val.split(",") if v.strip()]


class Config:
    # ---- Core Flask ---------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "")
    ENV = os.environ.get("FLASK_ENV", "production")
    DEBUG = _bool("FLASK_DEBUG", False)

    # ---- Database (DW_TT data warehouse) -------------------------------
    # Built from discrete POSTGRES_* vars by default (matches docker-compose),
    # or overridden wholesale with DATABASE_URL.
    DATABASE_URL = os.environ.get("DATABASE_URL") or (
        "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(
            user=os.environ.get("POSTGRES_USER", "postgres"),
            pw=os.environ.get("POSTGRES_PASSWORD", ""),
            host=os.environ.get("POSTGRES_HOST", "db"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            db=os.environ.get("POSTGRES_DB", "DW_TT"),
        )
    )

    # ---- CORS -----------------------------------------------------------
    CORS_ORIGINS = _list("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")

    # ---- Auth / JWT -------------------------------------------------------
    JWT_EXPIRES_MINUTES = int(os.environ.get("JWT_EXPIRES_MINUTES", "480"))  # 8h shift
    JWT_ALGORITHM = "HS256"

    # ---- ML pipeline ------------------------------------------------------
    # Re-run the full training/scoring pipeline against the DW on a timer.
    # 0 disables the timer entirely — refresh only happens on startup and via
    # POST /api/admin/refresh (admin only).
    ML_REFRESH_INTERVAL_MINUTES = int(os.environ.get("ML_REFRESH_INTERVAL_MINUTES", "0"))

    # Business parameters for the Objective 9 gain formula — configurable
    # rather than re-hardcoded, matching the 3 scenarios explored in the
    # original notebook (build_09.py).
    RETENTION_LTV_HORIZON_MONTHS = int(os.environ.get("RETENTION_LTV_HORIZON_MONTHS", "12"))
    RETENTION_OFFER_COST_TND = float(os.environ.get("RETENTION_OFFER_COST_TND", "15.0"))

    # ---- RBAC ---------------------------------------------------------
    # Which navigation sections / API scopes each role may access. Kept as
    # plain data (not code) so it can be edited later without touching the
    # route logic — see README "Roles and permissions".
    #
    # "admin" implicitly has every scope; ROLE_PERMISSIONS only needs to
    # describe non-admin roles.
    ROLE_PERMISSIONS: dict[str, list[str]] = {
        "user": [
            "dashboard",
            "clients",
            "churn",
            "segmentation",
            "retention",
            "revenue",
            "models",
        ],
    }

    # Full list of scopes, in nav order. "monitoring" and "settings" are
    # deliberately left out of the default "user" role above (operational /
    # admin concerns) but stay editable here for the operator.
    ALL_SCOPES = [
        "dashboard",
        "clients",
        "churn",
        "segmentation",
        "retention",
        "revenue",
        "models",
        "monitoring",
        "settings",
    ]


def get_config() -> type[Config]:
    return Config
