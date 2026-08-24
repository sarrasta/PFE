"""Churn & Risk page — Objectives 1 (classification), 2 (risk level) and 6
(continuous propensity), aggregated over the scored client base."""
from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.ml.pipeline import sanitize_record
from app.utils.responses import require_ready_pipeline

bp = Blueprint("churn", __name__, url_prefix="/api/churn")


def _histogram(series: pd.Series, bins: int = 20) -> list[dict]:
    values = series.dropna().values
    if len(values) == 0:
        return []
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    return [
        {"bucket_start": round(float(edges[i]), 3), "bucket_end": round(float(edges[i + 1]), 3),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]


@bp.get("/overview")
@login_required
@scope_required("churn")
@require_ready_pipeline
def overview():
    pipeline = extensions.ml_pipeline
    table = pipeline.client_table

    risk_dist = table["risk_label"].value_counts().to_dict()
    by_risk = (
        table.groupby("risk_label")[["churn_proba", "propensity_score", "avg_arpu"]]
        .mean().round(4).reset_index().to_dict(orient="records")
    )
    by_segment = (
        table.groupby("segment")["propensity_score"].mean().round(4).sort_values(ascending=False)
        .reset_index().to_dict(orient="records")
    )

    return jsonify(
        metrics=pipeline.metrics.get("objective_1_churn"),
        risk_metrics=pipeline.metrics.get("objective_2_risk"),
        propensity_metrics=pipeline.metrics.get("objective_6_propensity"),
        risk_distribution={k: int(v) for k, v in risk_dist.items()},
        churn_proba_histogram=_histogram(table["churn_proba"]),
        propensity_histogram=_histogram(table["propensity_score"]),
        profile_by_risk_level=by_risk,
        propensity_by_segment=by_segment,
        high_risk_count=int((table["risk_label"] == "Eleve").sum()),
    )


@bp.get("/top-risk")
@login_required
@scope_required("churn")
@require_ready_pipeline
def top_risk():
    """The N highest-propensity clients — feeds a "most urgent" widget."""
    limit = min(200, max(1, int(request.args.get("limit", 20))))
    table = extensions.ml_pipeline.client_table
    cols = ["client_id", "client_ref", "region", "segment", "risk_label",
            "churn_proba", "propensity_score", "avg_arpu", "gain_net"]
    top = table.sort_values("propensity_score", ascending=False).head(limit)[cols]
    return jsonify(items=[sanitize_record(r) for r in top.to_dict(orient="records")])
