"""
Objective 9 — Expected net financial gain of a retention action.

Formula from `ml pfe/build_09.py` (-> `09_retention_gain.ipynb`), unchanged:

    Gain_Brut(i) = P_churn(i) x P_reponse(i) x ARPU_predit(i) x LTV_horizon
    Gain_Net(i)  = Gain_Brut(i) - Cout_offre

Integration improvement over the original notebook: `build_09.py` could only
run standalone, so P_churn and ARPU_predit were each an inline proxy formula
re-derived from raw features (the report literally labels them "proxy Obj 6"
and "proxy Obj 7"). Because this application already keeps Objective 6's
calibrated propensity score and Objective 7's regression-predicted ARPU in
memory as part of the same pipeline run, this module uses those real,
upstream model outputs directly instead of recomputing the proxies — exactly
the "court terme" improvement RAPPORT_PFE_FINAL.md itself recommends
("Exporter les scores per-client depuis Obj 6 vers Obj 9"). P_reponse still
comes from Objective 8 as in the original. See README "Adaptations".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LTV_HORIZON_DEFAULT = 12
OFFER_COST_SCENARIOS = {"legere": 5.0, "standard": 15.0, "premium": 30.0}


def compute_gain(
    client_table: pd.DataFrame,
    ltv_horizon_months: int = LTV_HORIZON_DEFAULT,
    offer_cost: float = OFFER_COST_SCENARIOS["standard"],
) -> tuple[pd.DataFrame, dict]:
    """`client_table` must carry: client_id, propensity_score, response_proba,
    arpu_futur_predit. Returns (dataframe[client_id, gain_net, roi, cible], metrics).
    """
    df = client_table[["client_id", "propensity_score", "response_proba", "arpu_futur_predit"]].copy()
    df["arpu_predit"] = df["arpu_futur_predit"].clip(lower=1.0)
    df["prob_utile"] = df["propensity_score"] * df["response_proba"]
    df["clv_12m"] = df["arpu_predit"] * ltv_horizon_months
    df["gain_brut"] = df["prob_utile"] * df["clv_12m"]
    df["gain_net"] = df["gain_brut"] - offer_cost
    df["roi"] = df["gain_net"] / offer_cost
    df["cible"] = (df["gain_net"] > 0).astype(int)

    n = len(df)
    df_sorted = df.sort_values("gain_net", ascending=False).reset_index(drop=True)
    lift_at_pct = {}
    for pct in (10, 20, 30, 50):
        idx = max(0, int(n * pct / 100) - 1)
        gain_topk = float(df_sorted.iloc[: idx + 1]["gain_net"].clip(lower=0).sum())
        total_positive_gain = float(df["gain_net"].clip(lower=0).sum())
        random_share = (idx + 1) / n
        model_share = gain_topk / total_positive_gain if total_positive_gain > 0 else 0.0
        lift_at_pct[f"top_{pct}pct"] = round(model_share / random_share, 3) if random_share > 0 else 1.0

    mask_roi_pos = df["cible"] == 1
    scenarios = {}
    for label, cost in OFFER_COST_SCENARIOS.items():
        gain_net_s = df["gain_brut"] - cost
        mask_s = gain_net_s > 0
        scenarios[label] = {
            "offer_cost_tnd": cost,
            "n_roi_positive": int(mask_s.sum()),
            "pct_roi_positive": float(mask_s.mean()),
            "gain_total_tnd": round(float(gain_net_s[mask_s].sum()), 0),
        }

    metrics = {
        "formula": "P_churn(Obj6) x P_reponse(Obj8) x ARPU_predit(Obj7) x LTV_horizon - Cout_offre",
        "parameters": {"ltv_horizon_months": ltv_horizon_months, "offer_cost_tnd": offer_cost},
        "results": {
            "n_clients": n,
            "n_roi_positive": int(mask_roi_pos.sum()),
            "pct_roi_positive": float(mask_roi_pos.mean()),
            "gain_total_tnd": round(float(df.loc[mask_roi_pos, "gain_net"].sum()), 0),
            "gain_mean_tnd": round(float(df.loc[mask_roi_pos, "gain_net"].mean()), 1) if mask_roi_pos.any() else 0.0,
            "roi_mean": round(float(df.loc[mask_roi_pos, "roi"].mean()), 2) if mask_roi_pos.any() else 0.0,
        },
        "lift_at_pct": lift_at_pct,
        "scenarios": scenarios,
    }
    return df[["client_id", "gain_brut", "gain_net", "roi", "cible"]], metrics
