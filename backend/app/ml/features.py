"""Shared feature-engineering helpers, factored out of the near-identical
code that was copy-pasted across `01_churn_v5/v6`, `02_risk_classification`,
`04_segmentation_v3` and `05_retention_profiles` in the original notebooks.
The formulas themselves are unchanged — only the duplication is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["sexe", "client_segment", "type_abo", "region", "offre_actuelle", "offre_cible"]


def encode_categoricals(df: pd.DataFrame, columns: list[str] = CATEGORICAL_COLS, suffix: str = "") -> pd.DataFrame:
    """Label-encodes each categorical column in place (or into `{col}{suffix}`
    if a suffix is given), matching `LabelEncoder().fit_transform(...)` used
    throughout the notebooks. Encoders are fit fresh each pipeline refresh —
    there is no cross-request encoder drift because every objective always
    (re)encodes its own full snapshot in one shot.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        target = f"{col}{suffix}" if suffix else col
        out[target] = LabelEncoder().fit_transform(out[col].astype(str).fillna("UNK"))
    return out


def add_risk_composites(df: pd.DataFrame) -> pd.DataFrame:
    """The 6 engineered "risk" features reused verbatim across Objectives
    1, 2 and 5 (`score_risque_paiement`, `ratio_hors_forfait`, ...)."""
    out = df.copy()
    out["score_risque_paiement"] = out["total_impayes"] * (out["max_retard_paiement"] + 1)
    out["ratio_hors_forfait"] = out["avg_hors_forfait"] / (out["avg_montant_facture"] + 0.01)
    out["score_degradation_reseau"] = out["avg_drop_rate"] * (out["avg_outage_min"] + 1)
    out["flag_hors_engagement"] = (out["engagement_restant"] == 0).astype(int)
    out["intensite_usage"] = out["avg_minutes"] / (out["avg_arpu"] + 0.01)
    out["score_insatisfaction"] = out["total_tickets"] / (out["avg_nps"] + 1)
    return out


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


def log_zscore(series: pd.Series) -> pd.Series:
    return zscore(np.log1p(np.clip(series, 0, None)))


def norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + 1e-9)
