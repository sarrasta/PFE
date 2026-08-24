"""
Objective 5 — Personalized retention micro-profiles (targeting matrix).

Adapted from `ml pfe/05_retention_profiles.ipynb`: the original notebook
crosses a behavioral segment x risk-level 2D matrix (4 x 3 = 12 micro-profiles)
and ranks cells by estimated ROI using
    revenue_at_risk = n_clients x churn_proba x avg_arpu x 12
    roi_net         = revenue_at_risk x retention_rate - n_clients x offer_cost x churn_proba
That formula is reused verbatim below.

Adaptation: the original notebook re-ran its own from-scratch K-Means
segmentation AND risk clustering just for this one matrix, producing THIRD
set of segment names ("Gros Consommateurs", "Clients Debiteurs", ...)
different from both Objective 2's risk labels and Objective 4's segment
names. Rebuilding a dashboard on three inconsistent taxonomies of the same
10,000 clients would be confusing and is not good software engineering, so
this module instead reuses the already-computed canonical segment (Objective
4 v3) and risk level (Objective 2) columns. The financial formulas and
campaign-assignment logic are unchanged; only the labels feeding them are
now shared with the rest of the app instead of recomputed a third time. See
README "Adaptations" for the full rationale and the segment name mapping.
"""
from __future__ import annotations

import pandas as pd

from app.ml.segmentation_model import SEGMENT_NAMES
from app.ml.risk_model import RISK_LABELS

OFFER_COST_DEFAULT = 30.0
RETENTION_RATE_DEFAULT = 0.40

# Maps each Objective-4 canonical segment to the closest Objective-5 campaign
# archetype from the original notebook, by dominant behavioral driver.
_CAMPAIGNS = {
    ("Fort Usage", "Eleve"): "Offre VIP + conseiller dedie",
    ("Fort Usage", "Moyen"): "Upsell + programme fidelite",
    ("Fort Usage", "Faible"): "Programme ambassadeur + cross-sell",
    ("Sensibles au Prix", "Eleve"): "Plan paiement urgence + offre retention",
    ("Sensibles au Prix", "Moyen"): "Alerte precoce + echeancier flexible",
    ("Sensibles au Prix", "Faible"): "Monitoring + newsletter personnalisee",
    ("Risque Reseau", "Eleve"): "Fix technique + offre dedomagement",
    ("Risque Reseau", "Moyen"): "Amelioration reseau + communication",
    ("Risque Reseau", "Faible"): "Newsletter + amelioration reseau",
    ("Clients Inactifs", "Eleve"): "Appel proactif + compensation SAV",
    ("Clients Inactifs", "Moyen"): "Suivi SAV proactif + NPS survey",
    ("Clients Inactifs", "Faible"): "Monitoring standard",
}


def build_targeting_matrix(
    client_table: pd.DataFrame,
    offer_cost: float = OFFER_COST_DEFAULT,
    retention_rate: float = RETENTION_RATE_DEFAULT,
) -> dict:
    """`client_table` must have columns: segment, risk_label, propensity_score, avg_arpu."""
    rows = []
    for seg in SEGMENT_NAMES:
        for risk in RISK_LABELS:
            sub = client_table[(client_table["segment"] == seg) & (client_table["risk_label"] == risk)]
            n = len(sub)
            if n == 0:
                continue
            churn_p = float(sub["propensity_score"].mean())
            arpu = float(sub["avg_arpu"].mean())
            revenue_at_risk = n * churn_p * arpu * 12
            revenue_saved = revenue_at_risk * retention_rate
            cost_total = n * offer_cost * churn_p
            roi_net = revenue_saved - cost_total
            rows.append({
                "segment": seg,
                "risk_label": risk,
                "n_clients": int(n),
                "churn_proba_mean": round(churn_p, 4),
                "arpu_mean": round(arpu, 1),
                "revenue_at_risk_tnd": round(revenue_at_risk, 0),
                "roi_net_tnd": round(roi_net, 0),
                "campaign": _CAMPAIGNS.get((seg, risk), "Monitoring standard"),
            })

    rows.sort(key=lambda r: r["roi_net_tnd"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["priority"] = i

    total_revenue_at_risk = sum(r["revenue_at_risk_tnd"] for r in rows)
    total_roi_net = sum(r["roi_net_tnd"] for r in rows)

    urgency_buckets = {"Rouge": 0, "Orange": 0, "Vert": 0}
    for r in rows:
        if r["risk_label"] == "Eleve":
            urgency_buckets["Rouge"] += r["n_clients"]
        elif r["risk_label"] == "Moyen":
            urgency_buckets["Orange"] += r["n_clients"]
        else:
            urgency_buckets["Vert"] += r["n_clients"]

    return {
        "micro_profiles": rows,
        "n_micro_profiles": len(rows),
        "total_revenue_at_risk_tnd": round(total_revenue_at_risk, 0),
        "total_roi_net_tnd": round(total_roi_net, 0),
        "retention_rate_assumed": retention_rate,
        "offer_cost_tnd": offer_cost,
        "urgency_buckets": urgency_buckets,
    }
