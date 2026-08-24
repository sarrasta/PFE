"""
Objective 7 — Future ARPU & monthly bill regression.

Faithful port of `ml pfe/build_07.py` (-> `07_arpu_prediction.ipynb`):
temporal split (DateFK 25-33 = 9 months of features -> DateFK 34-36 = 3-month
future target), ~97 vectorised features per client (mean/std/min/max, recent
3-month average, growth rate, linear trend per numeric column) plus static
client attributes, evaluated with 3-fold CV across 3 regressors. RandomForest
wins on both targets in the original run (R² 0.962 ARPU / 0.960 Facture).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler

from app.ml.data_loader import ARPU_TARGET_DATES, ARPU_TRAIN_DATES

NUM_COLS = ["arpu", "montant_facture", "minutes", "data_gb", "sms",
            "hors_forfait", "impayes", "retard", "qos", "drop_rate",
            "throughput", "outage", "nps"]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0.5
    if not mask.any():
        return 0.0
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _build_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df_past = monthly_df[monthly_df["mois"].isin(ARPU_TRAIN_DATES)].copy()
    df_future = monthly_df[monthly_df["mois"].isin(ARPU_TARGET_DATES)].copy()
    df_s = df_past.sort_values(["client_id", "mois"])

    df_stats = df_s.groupby("client_id")[NUM_COLS].agg(["mean", "std", "min", "max"]).fillna(0)
    df_stats.columns = [f"{c}_{s}" for c, s in df_stats.columns]

    df_recent = (df_s[df_s["mois"].isin(ARPU_TRAIN_DATES[-3:])]
                 .groupby("client_id")[NUM_COLS].mean().fillna(0)
                 .rename(columns={c: f"{c}_recent" for c in NUM_COLS}))
    df_early = (df_s[df_s["mois"].isin(ARPU_TRAIN_DATES[:3])]
                .groupby("client_id")[NUM_COLS].mean().fillna(0))
    growth = (df_recent[[f"{c}_recent" for c in NUM_COLS]].values /
              (df_early.values + 0.001) - 1)
    df_growth = pd.DataFrame(growth, index=df_early.index, columns=[f"{c}_growth" for c in NUM_COLS])

    nd = len(ARPU_TRAIN_DATES)
    x = np.arange(nd, dtype=float)
    x -= x.mean()
    ss_xx = (x ** 2).sum()
    trend_records = {}
    for col in NUM_COLS:
        pivot = (df_s.pivot_table(index="client_id", columns="mois", values=col, fill_value=0)
                 .reindex(columns=sorted(ARPU_TRAIN_DATES), fill_value=0))
        v = pivot.values.astype(float)
        y_bar = v.mean(axis=1, keepdims=True)
        trend_records[f"{col}_trend"] = pd.Series((x * (v - y_bar)).sum(axis=1) / ss_xx, index=pivot.index)
    df_trend = pd.DataFrame(trend_records)

    static_cols = ["client_id", "client_segment", "type_abo", "region",
                   "anciennete_mois", "engagement_restant", "prix_offre"]
    df_static = df_s.groupby("client_id")[static_cols[1:]].first().reset_index()

    df_feat = (df_stats.join(df_recent).join(df_growth).join(df_trend)
               .reset_index().merge(df_static, on="client_id"))

    df_targets = (df_future.groupby("client_id")
                  .agg(arpu_futur=("arpu", "mean"), facture_futur=("montant_facture", "mean"))
                  .reset_index())
    return df_feat.merge(df_targets, on="client_id", how="inner")


def predict_future_arpu(monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (dataframe[client_id, arpu_futur_predit, facture_futur_predit], metrics dict)."""
    df_model = _build_features(monthly_df)

    cat_cols = ["client_segment", "type_abo", "region"]
    df_enc = df_model.copy()
    for col in cat_cols:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str).fillna("UNK"))

    excl = {"client_id", "arpu_futur", "facture_futur"}
    feat_cols = [c for c in df_enc.columns if c not in excl and df_enc[c].dtype != object]

    X = df_enc[feat_cols].fillna(0).values.astype(float)
    y_arpu = df_enc["arpu_futur"].values
    y_facture = df_enc["facture_futur"].values

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)

    models = {
        "GBR": GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                                          subsample=0.8, min_samples_leaf=10, random_state=42),
        "RF": RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10,
                                     n_jobs=-1, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def run_cv(model, X, y):
        oof = np.zeros(len(y))
        for tr, va in kf.split(X):
            model.fit(X[tr], y[tr])
            oof[va] = model.predict(X[va])
        return (r2_score(y, oof), mean_absolute_error(y, oof),
                float(np.sqrt(mean_squared_error(y, oof))), _mape(y, oof))

    results: dict[str, dict] = {}
    for tgt, y in [("arpu", y_arpu), ("facture", y_facture)]:
        for name, model in models.items():
            r2, mae, rmse, mape = run_cv(model, X_sc, y)
            results.setdefault(name, {})[tgt] = {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

    best_arpu = max(results, key=lambda k: results[k]["arpu"]["r2"])
    best_facture = max(results, key=lambda k: results[k]["facture"]["r2"])

    final_arpu = models[best_arpu]
    final_arpu.fit(X_sc, y_arpu)
    final_facture = models[best_facture]
    final_facture.fit(X_sc, y_facture)

    out = pd.DataFrame({
        "client_id": df_enc["client_id"].values,
        "arpu_futur_predit": final_arpu.predict(X_sc),
        "facture_futur_predit": final_facture.predict(X_sc),
        "arpu_futur_observe": y_arpu,
        "facture_futur_observe": y_facture,
    })

    metrics = {
        "best_model_arpu": best_arpu,
        "best_model_facture": best_facture,
        "n_features": len(feat_cols),
        "n_clients": int(len(out)),
        "models": {name: {tgt: {k: float(v) for k, v in vals.items()} for tgt, vals in res.items()}
                   for name, res in results.items()},
    }
    return out, metrics
