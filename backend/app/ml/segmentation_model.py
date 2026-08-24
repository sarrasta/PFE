"""
Objective 4 — Behavioral segmentation (v3 "business-defined" scores).

Faithful port of `ml pfe/04_segmentation_v3.ipynb`, the final and best of 3
iterations (v1 raw features silhouette 0.1162 -> v2 composite scores 0.1997
-> v3 business-redefined scores 0.2324 — see RAPPORT_PFE_FINAL.md chapter 7).
K-Means (k=4) on 4 near-orthogonal composite scores (usage intensity, price
pressure, network risk, inactivity), clusters named via the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`) against their dominant
score, then a Random Forest is trained for deployment on new snapshots.

This "v3" segmentation is treated as the single canonical segmentation used
everywhere else in this application (Objective 5's targeting matrix, the
Dashboard, the Clients table) — the original project's Objective 5 notebook
re-derived a second, differently-named segmentation from scratch, which would
make a single dashboard internally inconsistent; see README "Adaptations".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from app.ml.features import encode_categoricals, log_zscore, norm01, zscore

SEGMENT_NAMES = ["Fort Usage", "Sensibles au Prix", "Risque Reseau", "Clients Inactifs"]

_SUP_FEATS_RAW = [
    "anciennete_mois", "engagement_restant", "rgpd_consent", "prix_offre",
    "avg_minutes", "avg_data_gb", "avg_sms", "avg_roaming_gb", "avg_nocturne_ratio",
    "avg_arpu", "avg_montant_facture", "avg_hors_forfait",
    "avg_qos", "avg_drop_rate", "avg_throughput", "avg_outage_min",
    "total_tickets", "avg_delai_resolution", "avg_nps", "nb_mois_observes",
    "total_impayes", "max_retard_paiement",
]


def _composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_hf"] = df["avg_hors_forfait"] / (df["avg_montant_facture"] + 0.01)

    usage = (log_zscore(df["avg_minutes"]) + log_zscore(df["avg_data_gb"]) +
             log_zscore(df["avg_sms"]) + log_zscore(df["avg_nocturne_ratio"])) / 4
    price = (log_zscore(df["avg_hors_forfait"]) + zscore(df["ratio_hf"]) +
             log_zscore(df["avg_montant_facture"])) / 3
    network = (zscore(df["avg_drop_rate"]) + log_zscore(df["avg_outage_min"]) -
               zscore(df["avg_qos"]) - zscore(df["avg_throughput"])) / 4
    inactivity = -(zscore(df["avg_arpu"]) + zscore(df["anciennete_mois"]) +
                   zscore(df["nb_mois_observes"]) + log_zscore(df["avg_minutes"])) / 4

    scores = pd.DataFrame({
        "usage_intensity": usage, "price_pressure": price,
        "network_risk": network, "inactivity": inactivity,
    })
    return pd.DataFrame(StandardScaler().fit_transform(scores), columns=scores.columns, index=df.index)


def segment_clients(all_clients_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = all_clients_df.copy()
    scores_std = _composite_scores(df)
    X = scores_std.values

    km = KMeans(n_clusters=4, random_state=42, n_init=50, max_iter=2000)
    df["cluster_raw"] = km.fit_predict(X)

    sil = silhouette_score(X, df["cluster_raw"], sample_size=min(3000, len(df)), random_state=42)
    db = davies_bouldin_score(X, df["cluster_raw"])
    ch = calinski_harabasz_score(X, df["cluster_raw"])

    scores_std["cluster"] = df["cluster_raw"]
    cents = scores_std.groupby("cluster")[["usage_intensity", "price_pressure",
                                            "network_risk", "inactivity"]].mean()
    mat = np.column_stack([
        norm01(cents["usage_intensity"].values), norm01(cents["price_pressure"].values),
        norm01(cents["network_risk"].values), norm01(cents["inactivity"].values),
    ])
    row_idx, col_idx = linear_sum_assignment(-mat)
    seg_map = {int(row_idx[i]): SEGMENT_NAMES[col_idx[i]] for i in range(4)}
    df["segment"] = df["cluster_raw"].map(seg_map)

    # Deployment classifier
    df_enc = encode_categoricals(df, suffix="_enc")
    sup_feats = (
        ["sexe_enc", "client_segment_enc", "type_abo_enc", "region_enc", "offre_actuelle_enc"]
        + _SUP_FEATS_RAW
    )
    avail = [f for f in sup_feats if f in df_enc.columns]
    Xs = df_enc[avail].fillna(0).values.astype(float)
    ys = df_enc["cluster_raw"].values

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=12,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(rf, Xs, ys, cv=skf, scoring="accuracy", n_jobs=-1)
    f1 = cross_val_score(rf, Xs, ys, cv=skf, scoring="f1_macro", n_jobs=-1)

    seg_counts = df["segment"].value_counts().to_dict()
    metrics = {
        "n_clients": int(len(df)),
        "k_final": 4,
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "segment_distribution": {k: int(v) for k, v in seg_counts.items()},
        "classifier_cv_accuracy": float(acc.mean()),
        "classifier_cv_accuracy_std": float(acc.std()),
        "classifier_cv_f1_macro": float(f1.mean()),
        "classifier_cv_f1_macro_std": float(f1.std()),
    }
    return df, metrics
