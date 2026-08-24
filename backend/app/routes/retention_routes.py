"""Retention Campaigns page — Objective 5 targeting matrix, Objective 8
response-probability distribution and Objective 9 expected gain / ROI."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.ml.gain_model import compute_gain
from app.ml.pipeline import sanitize_record
from app.utils.responses import require_ready_pipeline

bp = Blueprint("retention", __name__, url_prefix="/api/retention")


@bp.get("/targeting-matrix")
@login_required
@scope_required("retention")
@require_ready_pipeline
def targeting_matrix():
    return jsonify(**extensions.ml_pipeline.retention_matrix)


@bp.get("/gain-overview")
@login_required
@scope_required("retention")
@require_ready_pipeline
def gain_overview():
    pipeline = extensions.ml_pipeline
    table = pipeline.client_table

    top_by_gain = (
        table.sort_values("gain_net", ascending=False)
        .head(20)[["client_id", "client_ref", "segment", "risk_label", "region",
                    "propensity_score", "response_proba", "arpu_futur_predit", "gain_net", "roi"]]
    )

    return jsonify(
        metrics=pipeline.metrics.get("objective_9_gain"),
        response_metrics=pipeline.metrics.get("objective_8_response"),
        top_clients_by_gain=[sanitize_record(r) for r in top_by_gain.to_dict(orient="records")],
    )


@bp.post("/simulate-scenario")
@login_required
@scope_required("retention")
@require_ready_pipeline
def simulate_scenario():
    """Re-run the Objective-9 gain formula with a custom LTV horizon / offer
    cost, without re-training anything — lets a user explore "what if the
    offer cost 25 TND" without an admin-only full pipeline refresh."""
    pipeline = extensions.ml_pipeline
    body = request.get_json(silent=True) or {}

    try:
        ltv_horizon = int(body.get("ltv_horizon_months", pipeline.ltv_horizon_months))
        offer_cost = float(body.get("offer_cost_tnd", pipeline.offer_cost_tnd))
    except (TypeError, ValueError):
        return jsonify(error="bad_request", message="ltv_horizon_months and offer_cost_tnd must be numeric."), 400

    if not (1 <= ltv_horizon <= 60) or not (0 < offer_cost <= 1000):
        return jsonify(error="bad_request", message="Parameters out of accepted range."), 400

    _scored, metrics = compute_gain(pipeline.client_table, ltv_horizon_months=ltv_horizon, offer_cost=offer_cost)
    return jsonify(**metrics)
