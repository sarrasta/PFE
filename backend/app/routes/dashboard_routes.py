"""GET /api/dashboard — the KPI cards + charts on the Dashboard homepage."""
from __future__ import annotations

from flask import Blueprint, jsonify

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.utils.responses import require_ready_pipeline

bp = Blueprint("dashboard", __name__, url_prefix="/api")


@bp.get("/dashboard")
@login_required
@scope_required("dashboard")
@require_ready_pipeline
def dashboard():
    pipeline = extensions.ml_pipeline
    return jsonify(
        kpis=pipeline.dashboard_kpis,
        retention_summary={
            "total_revenue_at_risk_tnd": pipeline.retention_matrix.get("total_revenue_at_risk_tnd"),
            "total_roi_net_tnd": pipeline.retention_matrix.get("total_roi_net_tnd"),
            "urgency_buckets": pipeline.retention_matrix.get("urgency_buckets"),
        },
        last_updated=pipeline.last_updated.isoformat() if pipeline.last_updated else None,
    )
