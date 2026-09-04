"""Interactive single-customer inference from the persisted scored population."""
from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from app import extensions
from app.auth.decorators import login_required, scope_required
from app.utils.responses import require_ready_pipeline

bp = Blueprint("prediction", __name__, url_prefix="/api")

NUMERIC_FIELDS = {
    "anciennete_mois": (0, 600), "engagement_restant": (0, 60), "prix_offre": (0, 1000),
    "avg_minutes": (0, 100000), "avg_data_gb": (0, 10000), "avg_sms": (0, 100000),
    "avg_arpu": (0, 10000), "avg_montant_facture": (0, 10000), "avg_hors_forfait": (0, 10000),
    "total_impayes": (0, 100000), "max_retard_paiement": (0, 1000), "total_tickets": (0, 10000),
    "avg_nps": (-100, 100), "avg_qos": (0, 10), "avg_drop_rate": (0, 100), "avg_outage_min": (0, 100000),
}
CATEGORICAL_FIELDS = ("sexe", "client_segment", "type_abo", "region", "offre_actuelle")


def _weighted_label(values, weights):
    totals = {}
    for value, weight in zip(values, weights):
        totals[str(value)] = totals.get(str(value), 0.0) + float(weight)
    return max(totals, key=totals.get)


@bp.post("/predict/customer")
@login_required
@scope_required("models")
@require_ready_pipeline
def predict_customer():
    body = request.get_json(silent=True) or {}
    stats = body.get("stats") or {}
    history = body.get("monthly_history") or []
    if not isinstance(stats, dict):
        return jsonify(error="bad_request", message="stats must be an object."), 400

    clean = {}
    errors = {}
    for field, (low, high) in NUMERIC_FIELDS.items():
        try:
            value = float(stats.get(field, 0))
            if not np.isfinite(value) or value < low or value > high:
                raise ValueError
            clean[field] = value
        except (TypeError, ValueError):
            errors[field] = f"Valeur attendue entre {low} et {high}."
    if errors:
        return jsonify(error="validation_error", message="Certaines statistiques sont invalides.", fields=errors), 400

    table = extensions.ml_pipeline.client_table
    features = [field for field in NUMERIC_FIELDS if field in table.columns]
    matrix = table[features].fillna(0).astype(float)
    center = matrix.median()
    scale = matrix.std().replace(0, 1).fillna(1)
    query = pd.Series({field: clean[field] for field in features})
    distances = np.sqrt((((matrix - center) / scale - (query - center) / scale) ** 2).mean(axis=1))
    for field in CATEGORICAL_FIELDS:
        value = str(stats.get(field, "")).strip()
        if value and field in table.columns:
            distances += (table[field].fillna("").astype(str) != value).astype(float) * 0.35

    neighbor_count = min(35, len(table))
    nearest_idx = distances.nsmallest(neighbor_count).index
    neighbors = table.loc[nearest_idx]
    weights = 1 / (distances.loc[nearest_idx].to_numpy() + 0.08)
    weights = weights / weights.sum()

    def weighted(field):
        return float(np.average(neighbors[field].fillna(neighbors[field].median()), weights=weights))

    predicted_arpu = weighted("arpu_futur_predit")
    valid_history = []
    if isinstance(history, list):
        for item in history[-12:]:
            try:
                value = float(item)
                if np.isfinite(value) and value >= 0:
                    valid_history.append(value)
            except (TypeError, ValueError):
                pass
    if len(valid_history) >= 3:
        slope = float(np.polyfit(np.arange(len(valid_history)), valid_history, 1)[0])
        predicted_arpu = max(0.0, 0.65 * predicted_arpu + 0.35 * (valid_history[-1] + slope * 3))

    churn = weighted("churn_proba")
    propensity = weighted("propensity_score")
    response = weighted("response_proba")
    gain = weighted("gain_net")
    risk = _weighted_label(neighbors["risk_label"], weights)
    segment = _weighted_label(neighbors["segment"], weights)
    should_target = bool(gain > 0 and (risk == "Eleve" or propensity >= 0.5))

    return jsonify(result={
        "churn_probability": round(churn, 4), "risk_level": risk, "segment": segment,
        "propensity_score": round(propensity, 4), "predicted_arpu_tnd": round(predicted_arpu, 2),
        "offer_response_probability": round(response, 4), "expected_gain_tnd": round(gain, 2),
        "recommendation": "Action de rétention prioritaire" if should_target else "Suivi standard",
        "priority": "Haute" if should_target else "Normale",
    }, model={"method": "inférence sur profils scorés persistés", "reference_profiles": neighbor_count, "monthly_points_used": len(valid_history)})
