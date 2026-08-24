"""
Objective 1 — Churn prediction (binary classification) + Objective 3 —
business-metric evaluation of the same models.

Faithful port of `ml pfe/01_churn_v6_synthesis.ipynb` (final iteration: SMOTE
+ SVC-RBF, C=10, gamma=0.001 — AUC 0.7138 ± 0.114 in the original 5-fold CV,
the best of 6 progressively-refined iterations documented in
RAPPORT_PFE_FINAL.md chapter 4) for the deployed classifier, and of
`02_risk_classification.ipynb` cell 4 for the exact feature subset used when
scoring the full 10,000-client base (`CHURN_FEATS` / `prep_features`) — that
notebook is the one that actually took the Objective-1 model to deployment,
so its feature list is treated as canonical here.

`evaluate_business_metrics()` is a faithful port of `03_business_metrics.ipynb`
(Objective 3): same 4 candidate pipelines, same CV protocol, same
Recall@TopK / Lift@TopK / cost-benefit formulas, run against the same 422
labeled clients — used only for the Model Performance page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

from app.ml.features import add_risk_composites

# Exact deployment feature set from 02_risk_classification.ipynb (CHURN_FEATS)
CHURN_FEATS = [
    "sexe", "client_segment", "type_abo", "anciennete_mois", "engagement_restant",
    "rgpd_consent", "region", "offre_actuelle", "prix_offre", "offre_cible",
    "avg_minutes", "avg_data_gb", "avg_montant_facture", "avg_arpu", "avg_hors_forfait",
    "total_impayes", "max_retard_paiement", "avg_retard_paiement",
    "total_tickets", "avg_delai_resolution", "avg_nps", "avg_qos",
    "avg_drop_rate", "avg_throughput", "avg_outage_min", "nb_mois_observes",
]
_CATEGORICAL = ["sexe", "client_segment", "type_abo", "region", "offre_actuelle", "offre_cible"]


def _prep_features(df: pd.DataFrame) -> np.ndarray:
    d = df[CHURN_FEATS].copy().fillna(0)
    for c in _CATEGORICAL:
        d[c] = LabelEncoder().fit_transform(d[c].astype(str))
    d["score_risque_paiement"] = d["total_impayes"] * (d["max_retard_paiement"] + 1)
    d["ratio_hors_forfait"] = d["avg_hors_forfait"] / (d["avg_montant_facture"] + 0.01)
    d["score_degradation_reseau"] = d["avg_drop_rate"] * (d["avg_outage_min"] + 1)
    d["flag_hors_engagement"] = (d["engagement_restant"] == 0).astype(int)
    d["score_insatisfaction"] = d["total_tickets"] / (d["avg_nps"] + 1)
    return d.values.astype(float)


def _build_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(k_neighbors=5, random_state=42)),
        ("scaler", RobustScaler()),
        ("clf", SVC(kernel="rbf", C=10.0, gamma=0.001, probability=True, random_state=42)),
    ])


def score_all_clients(labeled_df: pd.DataFrame, all_clients_df: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """Trains the deployed SVC-RBF pipeline on the 422 labeled clients and
    scores every client in `all_clients_df`.

    Returns (churn_proba indexed by client_id, cv_auc_mean, cv_auc_std).
    """
    X_train = _prep_features(labeled_df)
    y_train = labeled_df["churn"].values.astype(int)

    pipe = _build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    X_all = _prep_features(all_clients_df)
    proba = pipe.predict_proba(X_all)[:, 1]

    result = pd.Series(proba, index=all_clients_df["client_id"].values, name="churn_proba")
    return result, float(cv_scores.mean()), float(cv_scores.std())


# ---------------------------------------------------------------------------
# Objective 3 — business-metric comparison of 4 candidate models, run on the
# full 35-feature set (matches 03_business_metrics.ipynb exactly).
# ---------------------------------------------------------------------------

def _full_feature_matrix(labeled_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = labeled_df.copy()
    for col in _CATEGORICAL:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str).fillna("UNK"))
    df = add_risk_composites(df)
    exclude = {"client_id", "churn"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0).values.astype(float)
    y = df["churn"].values.astype(int)
    return X, y


def _recall_at_topk(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> float:
    n_top = max(1, int(len(y_true) * k))
    idx_sorted = np.argsort(y_prob)[::-1]
    top_true = y_true[idx_sorted[:n_top]]
    return float(top_true.sum() / (y_true.sum() + 1e-9))


def _lift_at_topk(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> float:
    recall_model = _recall_at_topk(y_true, y_prob, k)
    return recall_model / k if k > 0 else 1.0


def evaluate_business_metrics(labeled_df: pd.DataFrame) -> dict:
    X, y = _full_feature_matrix(labeled_df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LR_SMOTE": ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=0.05, solver="saga",
                                        max_iter=2000, random_state=42)),
        ]),
        "SVC_RBF": ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("sc", RobustScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma=0.001, probability=True, random_state=42)),
        ]),
        "RF_balanced": ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("sc", RobustScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1)),
        ]),
        "GB_balanced": ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("sc", RobustScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                learning_rate=0.05, random_state=42)),
        ]),
    }

    cv_proba: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        cv_proba[name] = proba
        thresholds = np.linspace(0.05, 0.95, 100)
        f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = float(thresholds[int(np.argmax(f1s))])
        metrics[name] = {
            "auc_roc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "recall_top10": round(_recall_at_topk(y, proba, 0.10), 4),
            "lift_top10": round(_lift_at_topk(y, proba, 0.10), 4),
            "recall_top20": round(_recall_at_topk(y, proba, 0.20), 4),
            "lift_top20": round(_lift_at_topk(y, proba, 0.20), 4),
            "brier_score": round(float(brier_score_loss(y, proba)), 4),
            "f1_at_optimal": round(float(max(f1s)), 4),
            "threshold_optimal": round(best_t, 3),
        }

    # Cost-benefit analysis on the best-AUC model (matches the notebook,
    # which hardcodes SVC_RBF as "v5 best" — here we pick it dynamically by
    # AUC so the logic still holds if the ranking ever changes).
    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best_proba = cv_proba[best_name]

    arpu_monthly, months_saved, offer_cost, retention_rate = 35.0, 12, 30.0, 0.40
    revenue_saved = arpu_monthly * months_saved * retention_rate
    net_gain_tp = revenue_saved - offer_cost
    net_loss_fp = -offer_cost

    thresholds = np.linspace(0.01, 0.99, 200)
    net_values, precisions, recalls, n_contacted = [], [], [], []
    for t in thresholds:
        y_pred = (best_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        net_values.append(tp * net_gain_tp + fp * net_loss_fp)
        precisions.append(tp / (tp + fp + 1e-9))
        recalls.append(tp / (tp + fn + 1e-9))
        n_contacted.append(int(tp + fp))
    net_values = np.array(net_values)
    opt_idx = int(np.argmax(net_values))

    return {
        "n_samples": int(len(y)),
        "prevalence": float(y.mean()),
        "models": metrics,
        "best_model": best_name,
        "cost_benefit": {
            "model_used": best_name,
            "optimal_threshold": round(float(thresholds[opt_idx]), 4),
            "precision": round(float(precisions[opt_idx]), 4),
            "recall": round(float(recalls[opt_idx]), 4),
            "net_value_tnd": round(float(net_values[opt_idx]), 1),
            "n_contacted": n_contacted[opt_idx],
            "breakeven_threshold": round(offer_cost / (net_gain_tp + offer_cost), 4),
            "business_params": {
                "arpu_monthly_tnd": arpu_monthly,
                "months_saved": months_saved,
                "retention_rate": retention_rate,
                "offer_cost_tnd": offer_cost,
            },
        },
    }
