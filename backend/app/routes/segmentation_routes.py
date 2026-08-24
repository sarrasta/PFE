"""Segmentation page — Objective 4 behavioral segmentation (v3, canonical)."""
from __future__ import annotations

from flask import Blueprint, jsonify

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.ml.segmentation_model import SEGMENT_NAMES
from app.utils.responses import require_ready_pipeline

bp = Blueprint("segmentation", __name__, url_prefix="/api/segmentation")

_PROFILE_COLS = [
    "avg_minutes", "avg_data_gb", "avg_sms", "avg_arpu", "avg_montant_facture",
    "avg_hors_forfait", "avg_qos", "avg_drop_rate", "avg_outage_min",
    "anciennete_mois", "nb_mois_observes", "churn_proba", "propensity_score",
]


@bp.get("/overview")
@login_required
@scope_required("segmentation")
@require_ready_pipeline
def overview():
    pipeline = extensions.ml_pipeline
    table = pipeline.client_table

    profile = (
        table.groupby("segment")[_PROFILE_COLS].mean().round(2)
        .reindex(SEGMENT_NAMES).reset_index().to_dict(orient="records")
    )
    counts = table["segment"].value_counts().reindex(SEGMENT_NAMES).fillna(0).astype(int).to_dict()
    by_region = (
        table.groupby(["region", "segment"]).size().unstack(fill_value=0)
        .reindex(columns=SEGMENT_NAMES, fill_value=0).reset_index().to_dict(orient="records")
    )

    return jsonify(
        metrics=pipeline.metrics.get("objective_4_segmentation"),
        segment_names=SEGMENT_NAMES,
        segment_counts=counts,
        segment_profile=profile,
        segment_by_region=by_region,
    )
