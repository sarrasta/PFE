"""Revenue & ARPU page — Objective 7 (future ARPU / bill regression)."""
from __future__ import annotations

from flask import Blueprint, jsonify

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.utils.responses import require_ready_pipeline

bp = Blueprint("revenue", __name__, url_prefix="/api/revenue")


@bp.get("/overview")
@login_required
@scope_required("revenue")
@require_ready_pipeline
def overview():
    pipeline = extensions.ml_pipeline
    table = pipeline.client_table

    by_segment = (
        table.groupby("segment")[["avg_arpu", "arpu_futur_predit", "avg_montant_facture",
                                   "facture_futur_predit"]]
        .mean().round(2).reset_index().to_dict(orient="records")
    )
    table = table.copy()
    table["arpu_delta"] = table["arpu_futur_predit"] - table["avg_arpu"]

    return jsonify(
        metrics=pipeline.metrics.get("objective_7_arpu"),
        arpu_current_mean=round(float(table["avg_arpu"].mean()), 2),
        arpu_predicted_mean=round(float(table["arpu_futur_predit"].mean()), 2),
        arpu_by_segment=by_segment,
        top_arpu_growth=[
            {"client_id": int(r.client_id), "region": r.region, "segment": r.segment,
             "avg_arpu": round(float(r.avg_arpu), 2), "arpu_futur_predit": round(float(r.arpu_futur_predit), 2),
             "arpu_delta": round(float(r.arpu_delta), 2)}
            for r in table.sort_values("arpu_delta", ascending=False).head(15).itertuples()
        ],
        top_arpu_decline=[
            {"client_id": int(r.client_id), "region": r.region, "segment": r.segment,
             "avg_arpu": round(float(r.avg_arpu), 2), "arpu_futur_predit": round(float(r.arpu_futur_predit), 2),
             "arpu_delta": round(float(r.arpu_delta), 2)}
            for r in table.sort_values("arpu_delta", ascending=True).head(15).itertuples()
        ],
    )
