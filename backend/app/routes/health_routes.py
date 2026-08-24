"""Public health check — no auth required, matches section 5/15 of the brief."""
from __future__ import annotations

import datetime as dt

from flask import Blueprint, jsonify

from app import extensions

bp = Blueprint("health", __name__, url_prefix="/api")


@bp.get("/health")
def health():
    pipeline = extensions.ml_pipeline
    store = extensions.user_store

    pipeline_info = {
        "status": pipeline.status if pipeline else "not_initialized",
        "last_updated": pipeline.last_updated.isoformat() if pipeline and pipeline.last_updated else None,
        "last_duration_seconds": pipeline.last_duration_seconds if pipeline else None,
        "clients_scored": int(len(pipeline.client_table)) if pipeline and pipeline.client_table is not None else 0,
    }
    if pipeline and pipeline.status == "error":
        pipeline_info["error"] = pipeline.error_message

    overall_ok = pipeline is not None and pipeline.status in ("ready", "loading")

    return jsonify(
        status="ok" if overall_ok else "degraded",
        service="tunisie-telecom-retention-analytics-backend",
        time=dt.datetime.now(dt.timezone.utc).isoformat(),
        ml_pipeline=pipeline_info,
        auth_configured=bool(store and store.is_configured()),
    ), 200 if overall_ok else 503
