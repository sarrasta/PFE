"""Model Performance page — every objective's cross-validated metrics in one
place, computed fresh on each pipeline run (not hardcoded)."""
from __future__ import annotations

from flask import Blueprint, jsonify

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.utils.responses import require_ready_pipeline

bp = Blueprint("models", __name__, url_prefix="/api/models")

_CATALOG = [
    {"id": "objective_1_churn", "name": "Prediction du churn", "type": "Classification",
     "headline_metric": "cv_auc_mean"},
    {"id": "objective_2_risk", "name": "Classification du risque", "type": "Clustering + Classification",
     "headline_metric": "rf_cv_accuracy"},
    {"id": "objective_3_business_metrics", "name": "Metriques metier", "type": "Evaluation",
     "headline_metric": None},
    {"id": "objective_4_segmentation", "name": "Segmentation comportementale", "type": "Clustering",
     "headline_metric": "silhouette"},
    {"id": "objective_5_retention", "name": "Profils de retention", "type": "Matrice de ciblage",
     "headline_metric": "total_roi_net_tnd"},
    {"id": "objective_6_propensity", "name": "Score de propension", "type": "Regression",
     "headline_metric": None},
    {"id": "objective_7_arpu", "name": "Prediction ARPU futur", "type": "Regression",
     "headline_metric": None},
    {"id": "objective_8_response", "name": "Reponse a une offre", "type": "Classification calibree",
     "headline_metric": None},
    {"id": "objective_9_gain", "name": "Gain attendu de retention", "type": "Formule analytique",
     "headline_metric": None},
]


@bp.get("")
@login_required
@scope_required("models")
@require_ready_pipeline
def list_models():
    pipeline = extensions.ml_pipeline
    items = []
    for entry in _CATALOG:
        items.append({**entry, "metrics": pipeline.metrics.get(entry["id"], {})})
    return jsonify(
        models=items,
        last_updated=pipeline.last_updated.isoformat() if pipeline.last_updated else None,
        last_duration_seconds=pipeline.last_duration_seconds,
    )


@bp.get("/<model_id>")
@login_required
@scope_required("models")
@require_ready_pipeline
def get_model(model_id: str):
    pipeline = extensions.ml_pipeline
    entry = next((e for e in _CATALOG if e["id"] == model_id), None)
    if entry is None:
        return jsonify(error="not_found", message=f"Unknown model id '{model_id}'."), 404
    return jsonify(**entry, metrics=pipeline.metrics.get(model_id, {}))
