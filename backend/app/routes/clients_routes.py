"""Client directory — search, filter, sort, paginate over the scored 10,000
clients, plus a single-client detail view (the "ML Predictions" drill-down:
every objective's output for one customer)."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.utils.responses import require_ready_pipeline

bp = Blueprint("clients", __name__, url_prefix="/api/clients")


@bp.get("")
@login_required
@scope_required("clients")
@require_ready_pipeline
def list_clients():
    pipeline = extensions.ml_pipeline
    args = request.args

    def _float(name):
        val = args.get(name)
        return float(val) if val not in (None, "") else None

    page = max(1, int(args.get("page", 1)))
    page_size = min(100, max(1, int(args.get("page_size", 25))))

    result = pipeline.query_clients(
        search=args.get("search") or None,
        risk_label=args.get("risk_label") or None,
        segment=args.get("segment") or None,
        region=args.get("region") or None,
        min_gain=_float("min_gain"),
        sort_by=args.get("sort_by", "gain_net"),
        sort_dir=args.get("sort_dir", "desc"),
        page=page,
        page_size=page_size,
    )
    return jsonify(**result)


@bp.get("/filters")
@login_required
@scope_required("clients")
@require_ready_pipeline
def filter_options():
    return jsonify(**extensions.ml_pipeline.filter_options())


@bp.get("/<int:client_id>")
@login_required
@scope_required("clients")
@require_ready_pipeline
def get_client(client_id: int):
    record = extensions.ml_pipeline.get_client(client_id)
    if record is None:
        return jsonify(error="not_found", message=f"No client with id {client_id}."), 404
    return jsonify(client=record)
