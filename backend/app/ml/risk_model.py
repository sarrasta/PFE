"""
Objective 2 — Risk level classification (Faible / Moyen / Eleve).

Faithful port of `ml pfe/02_risk_classification.ipynb`: unsupervised K-Means
(k=3) discovers the risk groups from 13 risk indicators (including the
Objective-1 `churn_proba`), clusters are ranked into Faible/Moyen/Eleve by a
composite risk score built from their centroids, and a Random Forest is then
trained on those discovered labels so new/updated client snapshots can be
classified directly (this is the same "unsupervised -> supervised" hand-off
the notebook itself uses; CV accuracy 97.34% in the original run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import add_risk_composites, encode_categoricals

RISK_FEATS = [
    "total_impayes", "max_retard_paiement", "avg_retard_paiement",
    "total_tickets", "avg_nps", "avg_qos", "avg_drop_rate",
    "avg_hors_forfait", "score_risque_paiement", "score_insatisfaction",
    "score_degradation_reseau", "flag_hors_engagement", "churn_proba",
]

_RISK_POS = ["total_impayes", "max_retard_paiement", "total_tickets",
             "score_risque_paiement", "score_insatisfaction",
             "score_degradation_reseau", "churn_proba"]
_RISK_NEG = ["avg_nps", "avg_qos"]

RISK_LABELS = ["Faible", "Moyen", "Eleve"]


def classify_risk(all_clients_df: pd.DataFrame, churn_proba: pd.Series) -> tuple[pd.DataFrame, dict]:
    """`all_clients_df` must already carry a `client_id` column. Returns
    (dataframe with cluster/risk_level/risk_label columns added, metrics dict).
    """
    df = add_risk_composites(all_clients_df.copy())
    df["churn_proba"] = df["client_id"].map(churn_proba)

    X_risk = df[RISK_FEATS].fillna(0).values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_risk)

    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    df["cluster"] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df["cluster"], sample_size=min(2000, len(df)), random_state=42)

    centers_raw = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=RISK_FEATS)
    pos_norm = (centers_raw[_RISK_POS] - centers_raw[_RISK_POS].min()) / (
        centers_raw[_RISK_POS].max() - centers_raw[_RISK_POS].min() + 1e-9)
    neg_norm = (centers_raw[_RISK_NEG] - centers_raw[_RISK_NEG].min()) / (
        centers_raw[_RISK_NEG].max() - centers_raw[_RISK_NEG].min() + 1e-9)
    composite = pos_norm.sum(axis=1) - neg_norm.sum(axis=1)
    composite_rank = composite.rank().astype(int)  # 1=lowest risk .. 3=highest
    cluster_to_risk = {cluster_id: int(rank) - 1 for cluster_id, rank in composite_rank.items()}

    df["risk_level"] = df["cluster"].map(cluster_to_risk)
    df["risk_label"] = df["risk_level"].map(dict(enumerate(RISK_LABELS)))

    # Supervised deployment classifier (Random Forest on discovered labels)
    df_enc = encode_categoricals(df, suffix="_enc")
    supervised_feats = [
        "sexe_enc", "client_segment_enc", "type_abo_enc", "anciennete_mois", "engagement_restant",
        "rgpd_consent", "region_enc", "offre_actuelle_enc", "prix_offre", "offre_cible_enc",
        "avg_minutes", "avg_data_gb", "avg_montant_facture", "avg_arpu", "avg_hors_forfait",
        "total_impayes", "max_retard_paiement", "avg_retard_paiement",
        "total_tickets", "avg_delai_resolution", "avg_nps", "avg_qos",
        "avg_drop_rate", "avg_throughput", "avg_outage_min", "nb_mois_observes",
        "score_risque_paiement", "ratio_hors_forfait", "score_degradation_reseau",
        "flag_hors_engagement", "score_insatisfaction", "churn_proba",
    ]
    avail_feats = [f for f in supervised_feats if f in df_enc.columns]
    X_sup = df_enc[avail_feats].fillna(0).values.astype(float)
    y_sup = df_enc["risk_level"].values.astype(int)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe_rf, X_sup, y_sup, cv=skf, scoring="accuracy")
    f1_scores = cross_val_score(pipe_rf, X_sup, y_sup, cv=skf, scoring="f1_macro")

    risk_distribution = df["risk_label"].value_counts().to_dict()
    mean_churn_by_risk = df.groupby("risk_label")["churn_proba"].mean().round(4).to_dict()

    metrics = {
        "n_clients_total": int(len(df)),
        "kmeans_k": 3,
        "silhouette_k3": float(sil),
        "risk_distribution": {k: int(v) for k, v in risk_distribution.items()},
        "mean_churn_proba_by_risk": {k: float(v) for k, v in mean_churn_by_risk.items()},
        "rf_cv_accuracy": float(acc_scores.mean()),
        "rf_cv_accuracy_std": float(acc_scores.std()),
        "rf_cv_f1_macro": float(f1_scores.mean()),
        "rf_cv_f1_macro_std": float(f1_scores.std()),
    }

    return df, metrics
