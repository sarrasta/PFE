"""
Objective 8 — Probability of responding to a retention offer.

Faithful port of `ml pfe/build_08.py` (-> `08_offer_response.ipynb`). The
DW_TT warehouse has no historical record of past retention-offer responses,
so the original project generates realistic synthetic labels from codified
business rules (high ARPU + tenure + satisfaction + moderate churn risk =
more likely to respond), with the logit intercept solved via Brent's method
so the label rate lands on exactly 30% — a documented, reproducible
generation protocol (see RAPPORT_PFE_FINAL.md chapter 11), not a hidden
fabrication: this is what "predicting offer response" means in a system that
has never run a retention campaign yet. Three calibrated classifiers are then
trained on those labels; LogReg wins by AUC in the original run (0.6185).

The `prop_proxy` computed here is used ONLY to build the synthetic training
labels for reproducibility with the original notebook. Everywhere else in
this application (dashboard, action matrix, Objective 9 gain formula), the
real Objective 6 propensity score is used instead of this proxy — matching
the "next steps" the report itself recommends (chapter "Perspectives -
court terme": wire Objective 6's score into downstream objectives instead of
recomputing an inline proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

NUM_COLS = ["arpu", "montant_facture", "minutes", "data_gb", "sms",
            "hors_forfait", "impayes", "retard", "qos", "drop_rate",
            "throughput", "outage", "nps"]


def _norm01(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def _build_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    all_dates = sorted(monthly_df["mois"].unique())
    recent_dates, early_dates = all_dates[-3:], all_dates[:3]

    df_s = monthly_df.copy()
    for c in NUM_COLS:
        df_s[c] = pd.to_numeric(df_s[c], errors="coerce").fillna(0)

    df_stats = df_s.groupby("client_id")[NUM_COLS].agg(["mean", "std", "min", "max"]).fillna(0)
    df_stats.columns = [f"{c}_{s}" for c, s in df_stats.columns]

    df_recent = (df_s[df_s["mois"].isin(recent_dates)].groupby("client_id")[NUM_COLS].mean()
                 .fillna(0).rename(columns={c: f"{c}_recent" for c in NUM_COLS}))
    df_early = (df_s[df_s["mois"].isin(early_dates)].groupby("client_id")[NUM_COLS].mean()
                .fillna(0).rename(columns={c: f"{c}_early" for c in NUM_COLS}))

    growth = {f"{c}_growth": (df_recent[f"{c}_recent"].values / (df_early[f"{c}_early"].values + 0.001) - 1)
              for c in NUM_COLS}
    df_growth = pd.DataFrame(growth, index=df_recent.index)

    nd = len(all_dates)
    x = np.arange(nd, dtype=float)
    x -= x.mean()
    ssx = (x ** 2).sum()
    trend = {}
    for col in NUM_COLS:
        piv = (df_s.pivot_table(index="client_id", columns="mois", values=col, fill_value=0)
               .reindex(columns=sorted(all_dates), fill_value=0))
        v = piv.values.astype(float)
        yb = v.mean(axis=1, keepdims=True)
        trend[f"{col}_trend"] = pd.Series((x * (v - yb)).sum(axis=1) / ssx, index=piv.index)
    df_trend = pd.DataFrame(trend)

    df_static = df_s.groupby("client_id")[["client_segment", "type_abo", "anciennete_mois",
                                            "engagement_restant", "prix_offre", "region"]].last()
    df_static["client_segment_enc"] = LabelEncoder().fit_transform(df_static["client_segment"].fillna("Inconnu"))
    df_static["type_abo_enc"] = LabelEncoder().fit_transform(df_static["type_abo"].fillna("Inconnu"))
    df_static["region_enc"] = LabelEncoder().fit_transform(df_static["region"].fillna("Inconnu"))
    for c in ("anciennete_mois", "engagement_restant"):
        df_static[c] = pd.to_numeric(df_static[c], errors="coerce").fillna(0)
    df_static["prix_offre"] = pd.to_numeric(df_static["prix_offre"], errors="coerce").fillna(25.0)

    static_cols = ["client_segment_enc", "type_abo_enc", "region_enc",
                   "anciennete_mois", "engagement_restant", "prix_offre"]

    df_feat = (df_stats.join(df_recent, how="left").join(df_early, how="left")
               .join(df_growth, how="left").join(df_trend, how="left")
               .join(df_static[static_cols], how="left").fillna(0))
    return df_feat


def _synthetic_labels(df_feat: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    arpu_n = _norm01(df_feat["arpu_mean"])
    anc_n = _norm01(df_feat["anciennete_mois"])
    qos_n = _norm01(df_feat["qos_mean"])
    drop_n = _norm01(df_feat["drop_rate_mean"])
    impayes_n = _norm01(df_feat["impayes_mean"])
    engag_n = _norm01(df_feat["engagement_restant"])
    minutes_n = _norm01(df_feat["minutes_mean"])
    nps_n = _norm01(df_feat["nps_mean"])

    prop_proxy = 1 / (1 + np.exp(-(
        -0.8 * arpu_n + 0.9 * impayes_n + 0.6 * drop_n - 0.5 * qos_n - 0.4 * nps_n
    )))
    prop_sweet = np.exp(-8.0 * (prop_proxy - 0.40) ** 2)

    logit_raw = (
        1.2 * arpu_n + 0.9 * anc_n + 0.7 * qos_n - 0.8 * drop_n - 0.9 * impayes_n
        + 1.4 * prop_sweet + 0.5 * minutes_n + 0.3 * (1 - engag_n) + 0.3 * nps_n
        + rng.normal(0, 0.45, len(df_feat))
    )

    target_rate = 0.30
    try:
        intercept = brentq(lambda t: (1 / (1 + np.exp(-(logit_raw + t)))).mean() - target_rate, -10, 10)
    except Exception:
        intercept = -1.0

    prob_true = 1 / (1 + np.exp(-(logit_raw + intercept)))
    y_response = rng.binomial(1, prob_true).astype(int)
    return y_response, prob_true.values, prop_proxy.values


def score_response(monthly_df: pd.DataFrame) -> tuple[pd.Series, dict]:
    df_feat = _build_features(monthly_df)
    y_response, _prob_true, _prop_proxy = _synthetic_labels(df_feat)

    feat_cols = list(df_feat.columns)
    X = df_feat[feat_cols].values.astype(float)

    models = {
        "LogReg": Pipeline([
            ("sc", RobustScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "GBR": CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, min_samples_leaf=10, random_state=42),
            method="isotonic", cv=3),
        "RF": CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=15,
                                    n_jobs=-1, class_weight="balanced", random_state=42),
            method="isotonic", cv=3),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results: dict[str, dict] = {}
    for name, model in models.items():
        oof = cross_val_predict(model, X, y_response, cv=cv, method="predict_proba")[:, 1]
        prec_arr, rec_arr, thrs = precision_recall_curve(y_response, oof)
        f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
        opt_t = float(thrs[int(np.argmax(f1s[:-1]))]) if len(thrs) else 0.5
        y_pred = (oof >= opt_t).astype(int)
        results[name] = {
            "auc": float(roc_auc_score(y_response, oof)),
            "avg_precision": float(average_precision_score(y_response, oof)),
            "brier": float(brier_score_loss(y_response, oof)),
            "precision": float(precision_score(y_response, y_pred, zero_division=0)),
            "recall": float(recall_score(y_response, y_pred, zero_division=0)),
            "best_threshold": opt_t,
        }

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = models[best_name]
    best_model.fit(X, y_response)
    resp_proba = best_model.predict_proba(X)[:, 1]

    scores = pd.Series(resp_proba, index=df_feat.index.values, name="response_proba")

    metrics = {
        "best_model": best_name,
        "models": results,
        "deployment": {
            "n_clients": int(len(scores)),
            "synthetic_response_rate": float(y_response.mean()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        },
    }
    return scores, metrics
