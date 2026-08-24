"""Monitoring, Settings and admin-only operational endpoints."""
from __future__ import annotations

import datetime as dt

from flask import Blueprint, current_app, jsonify

from app import extensions
from app.auth.decorators import admin_required, login_required, scope_required

bp = Blueprint("admin", __name__, url_prefix="/api")

_START_TIME = dt.datetime.now(dt.timezone.utc)


@bp.get("/monitoring")
@login_required
@scope_required("monitoring")
def monitoring():
    pipeline = extensions.ml_pipeline
    uptime_seconds = (dt.datetime.now(dt.timezone.utc) - _START_TIME).total_seconds()
    return jsonify(
        service_uptime_seconds=round(uptime_seconds, 1),
        ml_pipeline={
            "status": pipeline.status if pipeline else "not_initialized",
            "last_updated": pipeline.last_updated.isoformat() if pipeline and pipeline.last_updated else None,
            "last_duration_seconds": pipeline.last_duration_seconds if pipeline else None,
            "error_message": pipeline.error_message if pipeline else None,
            "clients_scored": int(len(pipeline.client_table)) if pipeline and pipeline.client_table is not None else 0,
        },
        database={
            "row_counts": pipeline.db_row_counts if pipeline else {},
        },
    )


@bp.get("/settings")
@login_required
@scope_required("settings")
def settings():
    cfg = current_app.config
    pipeline = extensions.ml_pipeline
    return jsonify(
        role_permissions=cfg["ROLE_PERMISSIONS"],
        all_scopes=cfg["ALL_SCOPES"],
        ml_parameters={
            "retention_ltv_horizon_months": pipeline.ltv_horizon_months if pipeline else None,
            "retention_offer_cost_tnd": pipeline.offer_cost_tnd if pipeline else None,
            "auto_refresh_interval_minutes": cfg["ML_REFRESH_INTERVAL_MINUTES"],
        },
        users=[u for u in (extensions.user_store.list_public() if extensions.user_store else [])],
    )


@bp.post("/admin/refresh")
@login_required
@admin_required
def trigger_refresh():
    pipeline = extensions.ml_pipeline
    if pipeline is None:
        return jsonify(error="not_initialized", message="ML pipeline is not initialized."), 503
    if pipeline.status == "loading":
        return jsonify(message="A refresh is already in progress.", status=pipeline.status), 202
    pipeline.start_background_refresh()
    return jsonify(message="Pipeline refresh started.", status="loading"), 202


@bp.get("/admin/users")
@login_required
@admin_required
def list_users():
    store = extensions.user_store
    return jsonify(users=store.list_public() if store else [], configured=bool(store and store.is_configured()))
