"""
Orchestrates the 9 objectives into one coherent, cached scoring pipeline.

This module is the only thing `app/routes/*.py` talk to for ML results. It
owns the single lifecycle the brief asks for: load data from DW_TT once,
train every model once, cache the results in memory, and only recompute on
an explicit refresh (admin action or an optional timer) — never per request.

Because every step re-runs against Postgres each time, a run takes roughly
4-6 minutes on modest hardware (dominated by Objectives 7/8's per-column
temporal feature engineering over 120k monthly rows, each cross-validated
across several candidate models). It executes in a background thread so
`/api/health` responds immediately; only the very first run (no cached data
yet) makes ML-backed endpoints return HTTP 503 with a "warming up" message —
a later refresh keeps serving the last good result until the new one is
ready (see `require_ready_pipeline` in app/utils/responses.py).
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import logging
import threading
import traceback

import numpy as np
import pandas as pd

from app.ml import (
    arpu_model,
    churn_model,
    data_loader,
    gain_model,
    propensity_model,
    response_model,
    retention_profiles,
    risk_model,
    segmentation_model,
)

logger = logging.getLogger("ml_pipeline")

CLIENT_PROFILE_COLS = [
    "client_id", "client_ref", "sexe", "client_segment", "type_abo",
    "anciennete_mois", "engagement_restant", "rgpd_consent",
    "region", "gouvernorat", "offre_actuelle", "prix_offre", "offre_cible",
    "avg_minutes", "avg_data_gb", "avg_sms", "avg_arpu", "avg_montant_facture",
    "avg_hors_forfait", "total_impayes", "max_retard_paiement", "total_tickets",
    "avg_nps", "avg_qos", "avg_drop_rate", "avg_outage_min", "nb_mois_observes",
]


class MLPipeline:
    def __init__(self, database_url: str, ltv_horizon_months: int = 12, offer_cost_tnd: float = 15.0, artifact_path: str | None = None):
        self.database_url = database_url
        self.ltv_horizon_months = ltv_horizon_months
        self.offer_cost_tnd = offer_cost_tnd
        self.artifact_path = artifact_path

        self._refresh_lock = threading.RLock()
        self.status: str = "idle"  # idle -> loading -> ready | error
        self.error_message: str | None = None
        self.last_updated: dt.datetime | None = None
        self.last_duration_seconds: float | None = None

        self.client_table: pd.DataFrame | None = None
        self.metrics: dict = {}
        self.dashboard_kpis: dict = {}
        self.retention_matrix: dict = {}
        self.db_row_counts: dict = {}

    # ------------------------------------------------------------------
    def load_artifact(self) -> bool:
        """Restore the last successful scoring run from trusted local storage."""
        if not self.artifact_path or not os.path.isfile(self.artifact_path):
            return False
        try:
            with open(self.artifact_path, "rb") as handle:
                artifact = pickle.load(handle)
            if artifact.get("version") != 1:
                raise ValueError("unsupported artifact version")
            if artifact.get("parameters") != {"ltv_horizon_months": self.ltv_horizon_months, "offer_cost_tnd": self.offer_cost_tnd}:
                raise ValueError("artifact parameters do not match configuration")
            self.client_table = artifact["client_table"]
            self.metrics = artifact["metrics"]
            self.dashboard_kpis = artifact["dashboard_kpis"]
            self.retention_matrix = artifact["retention_matrix"]
            self.db_row_counts = artifact["db_row_counts"]
            self.last_updated = artifact["last_updated"]
            self.last_duration_seconds = artifact.get("last_duration_seconds")
            self.status = "ready"
            self.error_message = None
            logger.info("Loaded persistent ML artifact from %s (%d clients).", self.artifact_path, len(self.client_table))
            return True
        except Exception as exc:
            logger.warning("Could not load ML artifact %s: %s; training once.", self.artifact_path, exc)
            return False

    def _save_artifact(self) -> None:
        if not self.artifact_path:
            return
        artifact = {
            "version": 1,
            "parameters": {"ltv_horizon_months": self.ltv_horizon_months, "offer_cost_tnd": self.offer_cost_tnd},
            "client_table": self.client_table, "metrics": self.metrics,
            "dashboard_kpis": self.dashboard_kpis, "retention_matrix": self.retention_matrix,
            "db_row_counts": self.db_row_counts, "last_updated": self.last_updated,
            "last_duration_seconds": self.last_duration_seconds,
        }
        directory = os.path.dirname(self.artifact_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self.artifact_path}.tmp"
        with open(temporary_path, "wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.artifact_path)
        logger.info("Saved persistent ML artifact to %s.", self.artifact_path)

    # ------------------------------------------------------------------
    def start_background_refresh(self) -> None:
        thread = threading.Thread(target=self.refresh, name="ml-pipeline-refresh", daemon=True)
        thread.start()

    def refresh(self) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            logger.info("Refresh already in progress — skipping concurrent call.")
            return
        try:
            self.status = "loading"
            self.error_message = None
            started = dt.datetime.now(dt.timezone.utc)
            logger.info("ML pipeline refresh starting...")

            engine = data_loader.get_engine(self.database_url)
            db_counts = data_loader.database_stats(engine)

            labeled_df = data_loader.load_labeled_clients(engine)
            all_df = data_loader.load_all_clients(engine)
            monthly_df = data_loader.load_monthly_performance(engine)
            logger.info("Loaded data: labeled=%s all=%s monthly=%s",
                        labeled_df.shape, all_df.shape, monthly_df.shape)

            # Objective 1 — churn classification, deployed to all clients
            churn_proba, churn_auc_mean, churn_auc_std = churn_model.score_all_clients(labeled_df, all_df)

            # Objective 3 — business-metric comparison (uses labeled set directly)
            business_metrics = churn_model.evaluate_business_metrics(labeled_df)

            # Objective 2 — risk clustering + supervised classifier
            risk_df, risk_metrics = risk_model.classify_risk(all_df, churn_proba)

            # Objective 4 — behavioral segmentation (canonical, v3)
            seg_df, seg_metrics = segmentation_model.segment_clients(all_df)

            # Objective 6 — continuous propensity score
            propensity_scores, propensity_metrics = propensity_model.score_propensity(labeled_df, all_df)

            # Objective 7 — future ARPU / bill regression
            arpu_df, arpu_metrics = arpu_model.predict_future_arpu(monthly_df)

            # Objective 8 — offer response probability
            response_scores, response_metrics = response_model.score_response(monthly_df)

            # ---- Merge into one master client table --------------------
            table = all_df[[c for c in CLIENT_PROFILE_COLS if c in all_df.columns]].copy()
            table["churn_proba"] = table["client_id"].map(churn_proba)
            table["risk_level"] = table["client_id"].map(risk_df.set_index("client_id")["risk_level"])
            table["risk_label"] = table["client_id"].map(risk_df.set_index("client_id")["risk_label"])
            table["segment"] = table["client_id"].map(seg_df.set_index("client_id")["segment"])
            table["propensity_score"] = table["client_id"].map(propensity_scores)
            table["response_proba"] = table["client_id"].map(response_scores)
            table = table.merge(
                arpu_df[["client_id", "arpu_futur_predit", "facture_futur_predit"]],
                on="client_id", how="left",
            )

            n_incomplete = int(table[["churn_proba", "risk_label", "segment", "propensity_score",
                                       "response_proba", "arpu_futur_predit"]].isna().any(axis=1).sum())
            if n_incomplete:
                logger.warning("%d/%d clients missing at least one score after merge.", n_incomplete, len(table))
                # Neutral fallbacks so the dashboard never crashes on a partial row —
                # these clients are still flagged via `data_complete` for the UI.
                table["arpu_futur_predit"] = table["arpu_futur_predit"].fillna(table["avg_arpu"])
                table["facture_futur_predit"] = table["facture_futur_predit"].fillna(table["avg_montant_facture"])
                table["response_proba"] = table["response_proba"].fillna(table["response_proba"].median())
                table["propensity_score"] = table["propensity_score"].fillna(table["propensity_score"].median())
            table["data_complete"] = ~table[["churn_proba", "risk_label", "segment"]].isna().any(axis=1)

            # Objective 9 — expected retention gain (uses Obj 6 + Obj 7 + Obj 8 directly)
            gain_df, gain_metrics = gain_model.compute_gain(
                table, ltv_horizon_months=self.ltv_horizon_months, offer_cost=self.offer_cost_tnd
            )
            table = table.merge(gain_df, on="client_id", how="left")

            # Objective 5 — retention targeting matrix (canonical segment x risk)
            retention_matrix = retention_profiles.build_targeting_matrix(table)

            dashboard_kpis = self._build_dashboard_kpis(table, db_counts)

            metrics = {
                "objective_1_churn": {
                    "model": "SVC-RBF (C=10, gamma=0.001) + SMOTE",
                    "cv_auc_mean": churn_auc_mean,
                    "cv_auc_std": churn_auc_std,
                },
                "objective_2_risk": risk_metrics,
                "objective_3_business_metrics": business_metrics,
                "objective_4_segmentation": seg_metrics,
                "objective_5_retention": {
                    "n_micro_profiles": retention_matrix["n_micro_profiles"],
                    "total_roi_net_tnd": retention_matrix["total_roi_net_tnd"],
                },
                "objective_6_propensity": propensity_metrics,
                "objective_7_arpu": arpu_metrics,
                "objective_8_response": response_metrics,
                "objective_9_gain": gain_metrics,
            }

            # ---- Atomic publish -----------------------------------------
            self.client_table = table
            self.metrics = metrics
            self.dashboard_kpis = dashboard_kpis
            self.retention_matrix = retention_matrix
            self.db_row_counts = db_counts
            self.last_updated = dt.datetime.now(dt.timezone.utc)
            self.last_duration_seconds = (self.last_updated - started).total_seconds()
            self.status = "ready"
            self._save_artifact()
            logger.info("ML pipeline refresh complete in %.1fs (%d clients).",
                        self.last_duration_seconds, len(table))

        except Exception as exc:  # noqa: BLE001 — surfaced via /api/health & /api/admin
            logger.error("ML pipeline refresh failed: %s\n%s", exc, traceback.format_exc())
            self.status = "error"
            self.error_message = str(exc)
        finally:
            self._refresh_lock.release()

    # ------------------------------------------------------------------
    @staticmethod
    def _build_dashboard_kpis(table: pd.DataFrame, db_counts: dict) -> dict:
        risk_dist = table["risk_label"].value_counts().to_dict()
        segment_dist = table["segment"].value_counts().to_dict()
        n = len(table)

        return {
            "n_clients_total": int(n),
            "n_clients_scored": int(table["data_complete"].sum()),
            "avg_churn_proba": round(float(table["churn_proba"].mean()), 4),
            "avg_propensity_score": round(float(table["propensity_score"].mean()), 4),
            "avg_arpu_current": round(float(table["avg_arpu"].mean()), 2),
            "avg_arpu_predicted": round(float(table["arpu_futur_predit"].mean()), 2),
            "risk_distribution": {k: int(v) for k, v in risk_dist.items()},
            "risk_distribution_pct": {k: round(float(v) / n * 100, 1) for k, v in risk_dist.items()},
            "segment_distribution": {k: int(v) for k, v in segment_dist.items()},
            "high_risk_clients": int((table["risk_label"] == "Eleve").sum()),
            "total_revenue_at_risk_tnd": round(
                float((table["avg_arpu"] * table["churn_proba"] * 12).sum()), 0
            ),
            "total_expected_gain_tnd": round(float(table.loc[table["cible"] == 1, "gain_net"].sum()), 0),
            "n_roi_positive_clients": int((table["cible"] == 1).sum()),
            "database_row_counts": db_counts,
        }

    # ------------------------------------------------------------------
    def get_client(self, client_id: int) -> dict | None:
        if self.client_table is None:
            return None
        row = self.client_table[self.client_table["client_id"] == client_id]
        if row.empty:
            return None
        record = row.iloc[0].to_dict()
        return sanitize_record(record)

    def query_clients(
        self,
        search: str | None = None,
        risk_label: str | None = None,
        segment: str | None = None,
        region: str | None = None,
        min_gain: float | None = None,
        sort_by: str = "gain_net",
        sort_dir: str = "desc",
        page: int = 1,
        page_size: int = 25,
    ) -> dict:
        table = self.client_table
        if table is None:
            return {"items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}

        df = table
        if search:
            needle = search.strip()
            if needle.isdigit():
                df = df[(df["client_id"].astype(str) == needle) | (df["client_ref"].astype(str) == needle)]
            else:
                mask = (
                    df["region"].astype(str).str.contains(needle, case=False, na=False)
                    | df["gouvernorat"].astype(str).str.contains(needle, case=False, na=False)
                    | df["offre_actuelle"].astype(str).str.contains(needle, case=False, na=False)
                )
                df = df[mask]
        if risk_label:
            df = df[df["risk_label"] == risk_label]
        if segment:
            df = df[df["segment"] == segment]
        if region:
            df = df[df["region"] == region]
        if min_gain is not None:
            df = df[df["gain_net"] >= min_gain]

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=(sort_dir == "asc"), na_position="last")

        total = len(df)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        page_df = df.iloc[start: start + page_size]

        return {
            "items": [sanitize_record(r) for r in page_df.to_dict(orient="records")],
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    def filter_options(self) -> dict:
        if self.client_table is None:
            return {"risk_labels": [], "segments": [], "regions": []}
        table = self.client_table
        return {
            "risk_labels": sorted(x for x in table["risk_label"].dropna().unique().tolist()),
            "segments": sorted(x for x in table["segment"].dropna().unique().tolist()),
            "regions": sorted(x for x in table["region"].dropna().unique().tolist()),
        }


def sanitize_record(record: dict) -> dict:
    """Round floats and coerce numpy scalar types to plain Python for
    clean, compact JSON — the frontend never needs float64 precision."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
            out[k] = None
        else:
            out[k] = v
    return out
