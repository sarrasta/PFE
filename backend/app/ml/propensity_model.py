"""
Objective 6 — Continuous churn propensity score.

Faithful port of `ml pfe/build_06.py` (-> `06_churn_propensity.ipynb`):
regression (not classification) on the same binary label, trained with
class-balanced sample weights (minority non-churners upweighted 395/27x) so
the regressor doesn't collapse to predicting the 93.6% prevalence for every
client, followed by isotonic calibration fit on out-of-fold predictions.
Four candidate regressors are compared by out-of-fold AUC-ROC (ranking
quality) — Ridge wins in the original run (AUC 0.6772) because the
churn/feature relationship is largely linear in this normalized feature
space; tree-based models regress to ~0.5 for everyone (low Brier, useless
ranking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score

from app.ml.features import add_risk_composites

CAT_COLS = ["sexe", "client_segment", "type_abo", "region", "offre_actuelle", "offre_cible"]


def _engineer(
    df: pd.DataFrame,
    encoders: dict | None,
    scaler: RobustScaler | None,
    fit: bool,
    feat_cols: list[str] | None = None,
):
    """`labeled_df` (422 rows) and `all_clients_df` (10k rows) come from two
    separate SQL queries in data_loader.py whose SELECT lists are not in the
    same column order. Recomputing `feat_cols` from `d.columns` independently
    on each call would silently select the SAME POSITIONS but DIFFERENT
    FEATURES between train and score time (e.g. avg_sms/avg_data_gb swapped),
    corrupting every prediction — so at fit=False we always reuse the exact
    `feat_cols` list captured at fit time and select by name (`d[feat_cols]`),
    which is order-independent regardless of how the source dataframe's
    columns happen to be arranged.
    """
    d = df.copy()
    encs: dict = {} if encoders is None else encoders

    for col in CAT_COLS:
        if col not in d.columns:
            continue
        if fit:
            le = LabelEncoder()
            d[col] = le.fit_transform(d[col].astype(str).fillna("UNK"))
            encs[col] = le
        else:
            le = encs[col]
            known = set(le.classes_)
            d[col] = d[col].astype(str).fillna("UNK").apply(lambda x: x if x in known else le.classes_[0])
            d[col] = le.transform(d[col])

    for col in ["total_impayes", "max_retard_paiement", "avg_retard_paiement",
                "total_tickets", "avg_outage_min", "avg_hors_forfait"]:
        if col in d.columns:
            d[col] = np.log1p(np.clip(d[col], 0, None))

    d = add_risk_composites(d)

    if fit:
        exclude = {"client_id", "churn", "client_ref", "gouvernorat"}
        feat_cols = [c for c in d.columns if c not in exclude and d[c].dtype != object]
    X = d[feat_cols].fillna(0).values.astype(float)

    if fit:
        sc = RobustScaler()
        X = sc.fit_transform(X)
        return X, feat_cols, encs, sc
    else:
        X = scaler.transform(X)
        return X, feat_cols


def score_propensity(labeled_df: pd.DataFrame, all_clients_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    X_train, feat_cols, encoders, scaler = _engineer(labeled_df, None, None, fit=True)
    y_train = labeled_df["churn"].values.astype(float)

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    ratio = n_pos / max(n_neg, 1)
    sample_weights = np.where(y_train == 1, 1.0, float(ratio))

    models = {
        "GBR_balanced": GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.04,
                                                    subsample=0.8, min_samples_leaf=3, random_state=42),
        "RFR_balanced": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                               random_state=42),
        "Ridge": Ridge(alpha=5.0),
        "SVR_rbf": SVR(kernel="rbf", C=2.0, gamma="scale", epsilon=0.05),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict] = {}

    for name, model in models.items():
        oof_cal = np.zeros(len(y_train))
        for tr_idx, val_idx in skf.split(X_train, y_train):
            sw_tr = sample_weights[tr_idx]
            try:
                model.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=sw_tr)
            except TypeError:
                model.fit(X_train[tr_idx], y_train[tr_idx])
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(model.predict(X_train[tr_idx]), y_train[tr_idx], sample_weight=sample_weights[tr_idx])
            oof_cal[val_idx] = iso.predict(model.predict(X_train[val_idx]))
        oof_cal = np.clip(oof_cal, 0, 1)

        thresholds = np.linspace(0.05, 0.95, 200)
        accs = [accuracy_score(y_train, (oof_cal >= t).astype(int)) for t in thresholds]
        results[name] = {
            "auc": float(roc_auc_score(y_train, oof_cal)),
            "brier": float(brier_score_loss(y_train, oof_cal)),
            "r2": float(r2_score(y_train, oof_cal)),
            "mae": float(mean_absolute_error(y_train, oof_cal)),
            "best_accuracy": float(max(accs)),
            "best_threshold": float(thresholds[int(np.argmax(accs))]),
        }

    best_name = max(results, key=lambda k: results[k]["auc"])
    final_model = models[best_name]
    try:
        final_model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        final_model.fit(X_train, y_train)
    iso_final = IsotonicRegression(out_of_bounds="clip")
    iso_final.fit(final_model.predict(X_train), y_train, sample_weight=sample_weights)

    X_all, _ = _engineer(all_clients_df, encoders, scaler, fit=False, feat_cols=feat_cols)
    raw_scores = final_model.predict(X_all)
    propensity = np.clip(iso_final.predict(raw_scores), 0, 1)

    scores = pd.Series(propensity, index=all_clients_df["client_id"].values, name="propensity_score")

    metrics = {
        "best_model": best_name,
        "models": results,
        "deployment": {
            "n_clients": int(len(scores)),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        },
    }
    return scores, metrics
