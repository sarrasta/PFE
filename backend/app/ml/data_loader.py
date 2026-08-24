"""
Data access layer for the DW_TT (Data Warehouse Tunisie Telecom) PostgreSQL
star schema, restored from `ml pfe/datasql/todhia.backup` at container start.

Every query here is carried over from the original research notebooks
(`ml pfe/01_churn_v6_synthesis.ipynb`, `02_risk_classification.ipynb`,
`04_segmentation_v3.ipynb`, `05_retention_profiles.ipynb`, `build_07/08/09.py`)
almost verbatim — the only change is consolidating several near-duplicate
per-notebook SELECTs (each notebook re-wrote the same join by hand) into three
shared loaders reused by every objective module, so the join logic is defined
once instead of nine times.

Schema reminder (public schema, star layout):
  dim_client(Client_PK, ClientID, Sexe, Segment, TypeAbonnement, AncienneteMois,
             EngagementRestantMois, TelephoneHash, RGPD_Consent)
  dim_geographique(Geo_PK, ClientID, Region, Gouvernorat, Delegation, Localite)
  dim_offre(Offre_PK, ClientID, OffreActuelle, PrixOffre, OffreCible, Justification)
  Fact_performance_client(ClientFK, DateFK, GeoFK, OffreFK, Minutes, SMS, DataGB,
             RoamingGB, UsageNocturneRatio, MontantFacture, ARPU, HorsForfait,
             Impayes, RetardPaiementJours, Tickets, DelaiResolutionJ, NPS,
             QoSScore, DropRatePct, ThroughputMbps, OutageMinutes)
  Fact_churn(ClientFK, DateFK, Churn_30j, ChurnScore_30j)   — DateFK=36 labeled subset (422 rows)
  Fact_recommandation / Fact_prediction — not consumed here (empty in the source dump)

DateFK 25-36 = the 12 observed months; Fact_churn labels exist only at DateFK=36.
"""
from __future__ import annotations

import threading

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_engine: Engine | None = None
_engine_lock = threading.Lock()

CHURN_LABEL_DATEFK = 36
MONTHLY_DATE_RANGE = (25, 36)  # inclusive, matches build_07/08/09.py TRAIN/PRED windows
ARPU_TRAIN_DATES = list(range(25, 34))   # 9 months of features
ARPU_TARGET_DATES = list(range(34, 37))  # 3-month future target


def get_engine(database_url: str) -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)
        return _engine


def dispose_engine() -> None:
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


# ---------------------------------------------------------------------------
# Loader 1 — the 422 clients with an observed churn label (DateFK=36).
# Used by: churn_model (Obj 1), risk_model (Obj 2, to train the churn feature),
#          retention_profiles / propensity_model (Obj 6) training set.
# ---------------------------------------------------------------------------
_LABELED_QUERY = """
SELECT
    fc."ClientFK"              AS client_id,
    fc."Churn_30j"              AS churn,
    dc."Sexe"                   AS sexe,
    dc."Segment"                 AS client_segment,
    dc."TypeAbonnement"          AS type_abo,
    dc."AncienneteMois"          AS anciennete_mois,
    dc."EngagementRestantMois"   AS engagement_restant,
    CASE WHEN dc."RGPD_Consent" = 'True' THEN 1 ELSE 0 END AS rgpd_consent,
    COALESCE(dg."Region", 'Inconnu')       AS region,
    do_."OffreActuelle"          AS offre_actuelle,
    do_."PrixOffre"              AS prix_offre,
    COALESCE(do_."OffreCible", 'None')     AS offre_cible,
    AVG(fp."Minutes")            AS avg_minutes,
    AVG(fp."SMS")                AS avg_sms,
    AVG(fp."DataGB")             AS avg_data_gb,
    AVG(fp."RoamingGB")          AS avg_roaming_gb,
    AVG(fp."UsageNocturneRatio") AS avg_nocturne_ratio,
    AVG(fp."MontantFacture")     AS avg_montant_facture,
    AVG(fp."ARPU")               AS avg_arpu,
    AVG(fp."HorsForfait")        AS avg_hors_forfait,
    SUM(fp."Impayes")            AS total_impayes,
    MAX(fp."RetardPaiementJours") AS max_retard_paiement,
    AVG(fp."RetardPaiementJours") AS avg_retard_paiement,
    COUNT(fp."Tickets")          AS total_tickets,
    AVG(fp."DelaiResolutionJ")   AS avg_delai_resolution,
    AVG(fp."NPS")                AS avg_nps,
    AVG(fp."QoSScore")           AS avg_qos,
    AVG(fp."DropRatePct")        AS avg_drop_rate,
    AVG(fp."ThroughputMbps")     AS avg_throughput,
    AVG(fp."OutageMinutes")      AS avg_outage_min,
    COUNT(fp."DateFK")           AS nb_mois_observes
FROM public."Fact_churn" fc
INNER JOIN public."dim_client" dc ON fc."ClientFK" = dc."Client_PK"
LEFT  JOIN public."dim_geographique" dg ON dc."ClientID" = dg."ClientID"
LEFT  JOIN public."dim_offre" do_ ON dc."ClientID" = do_."ClientID"
LEFT  JOIN public."Fact_performance_client" fp ON fc."ClientFK" = fp."ClientFK"
WHERE fc."DateFK" = :datefk
GROUP BY fc."ClientFK", fc."Churn_30j",
    dc."Sexe", dc."Segment", dc."TypeAbonnement",
    dc."AncienneteMois", dc."EngagementRestantMois", dc."RGPD_Consent",
    dg."Region", do_."OffreActuelle", do_."PrixOffre", do_."OffreCible"
"""


def load_labeled_clients(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(text(_LABELED_QUERY), engine, params={"datefk": CHURN_LABEL_DATEFK})
    return df


# ---------------------------------------------------------------------------
# Loader 2 — all 10,000 clients, aggregated across their observed months,
# with the extra dims (Gouvernorat, client_ref) that Objectives 2/4/5 need
# for drill-down/crosstab views. This is the "current snapshot" every client
# is scored against.
# ---------------------------------------------------------------------------
_ALL_CLIENTS_QUERY = """
SELECT
    dc."Client_PK"                AS client_id,
    dc."ClientID"                  AS client_ref,
    dc."Sexe"                      AS sexe,
    dc."Segment"                    AS client_segment,
    dc."TypeAbonnement"             AS type_abo,
    dc."AncienneteMois"             AS anciennete_mois,
    dc."EngagementRestantMois"      AS engagement_restant,
    CASE WHEN dc."RGPD_Consent" = 'True' THEN 1 ELSE 0 END AS rgpd_consent,
    COALESCE(dg."Region", 'Inconnu')       AS region,
    COALESCE(dg."Gouvernorat", 'Inconnu')  AS gouvernorat,
    COALESCE(do_."OffreActuelle", 'None')  AS offre_actuelle,
    COALESCE(do_."PrixOffre", 0)            AS prix_offre,
    COALESCE(do_."OffreCible", 'None')     AS offre_cible,
    COALESCE(AVG(fp."Minutes"), 0)              AS avg_minutes,
    COALESCE(AVG(fp."DataGB"), 0)               AS avg_data_gb,
    COALESCE(AVG(fp."SMS"), 0)                  AS avg_sms,
    COALESCE(AVG(fp."RoamingGB"), 0)            AS avg_roaming_gb,
    COALESCE(AVG(fp."UsageNocturneRatio"), 0)   AS avg_nocturne_ratio,
    COALESCE(AVG(fp."MontantFacture"), 0)       AS avg_montant_facture,
    COALESCE(AVG(fp."ARPU"), 0)                 AS avg_arpu,
    COALESCE(AVG(fp."HorsForfait"), 0)          AS avg_hors_forfait,
    COALESCE(SUM(fp."Impayes"), 0)              AS total_impayes,
    COALESCE(MAX(fp."RetardPaiementJours"), 0)  AS max_retard_paiement,
    COALESCE(AVG(fp."RetardPaiementJours"), 0)  AS avg_retard_paiement,
    COALESCE(SUM(fp."Tickets"), 0)              AS total_tickets,
    COALESCE(AVG(fp."DelaiResolutionJ"), 0)     AS avg_delai_resolution,
    COALESCE(AVG(fp."NPS"), 5)                  AS avg_nps,
    COALESCE(AVG(fp."QoSScore"), 5)             AS avg_qos,
    COALESCE(AVG(fp."DropRatePct"), 0)          AS avg_drop_rate,
    COALESCE(AVG(fp."ThroughputMbps"), 0)       AS avg_throughput,
    COALESCE(AVG(fp."OutageMinutes"), 0)        AS avg_outage_min,
    COUNT(fp."DateFK")                          AS nb_mois_observes
FROM public."dim_client" dc
LEFT JOIN public."dim_geographique" dg ON dc."ClientID" = dg."ClientID"
LEFT JOIN public."dim_offre" do_ ON dc."ClientID" = do_."ClientID"
LEFT JOIN public."Fact_performance_client" fp ON dc."Client_PK" = fp."ClientFK"
GROUP BY dc."Client_PK", dc."ClientID", dc."Sexe", dc."Segment", dc."TypeAbonnement",
    dc."AncienneteMois", dc."EngagementRestantMois", dc."RGPD_Consent",
    dg."Region", dg."Gouvernorat", do_."OffreActuelle", do_."PrixOffre", do_."OffreCible"
ORDER BY dc."Client_PK"
"""


def load_all_clients(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(_ALL_CLIENTS_QUERY, engine)


# ---------------------------------------------------------------------------
# Loader 3 — raw month-level performance rows (DateFK 25-36) joined to the
# static client dimensions. Used by the temporal-feature objectives: ARPU
# regression (Obj 7), offer-response (Obj 8) and retention gain (Obj 9), each
# of which builds its own (mean/std/recent/growth/trend) feature matrix from
# this same monthly panel.
# ---------------------------------------------------------------------------
_MONTHLY_QUERY = """
SELECT
    fp."ClientFK"               AS client_id,
    fp."DateFK"                 AS mois,
    fp."ARPU"                   AS arpu,
    fp."MontantFacture"         AS montant_facture,
    fp."Minutes"                AS minutes,
    fp."DataGB"                 AS data_gb,
    fp."SMS"                    AS sms,
    fp."HorsForfait"            AS hors_forfait,
    fp."Impayes"                AS impayes,
    fp."RetardPaiementJours"    AS retard,
    fp."QoSScore"               AS qos,
    fp."DropRatePct"            AS drop_rate,
    fp."ThroughputMbps"         AS throughput,
    fp."OutageMinutes"          AS outage,
    fp."NPS"                    AS nps,
    dc."Segment"                AS client_segment,
    dc."TypeAbonnement"         AS type_abo,
    dc."AncienneteMois"         AS anciennete_mois,
    dc."EngagementRestantMois"  AS engagement_restant,
    COALESCE(do_."PrixOffre", 0)       AS prix_offre,
    COALESCE(dg."Region", 'Inconnu')   AS region
FROM public."Fact_performance_client" fp
INNER JOIN public."dim_client" dc ON fp."ClientFK" = dc."Client_PK"
LEFT  JOIN public."dim_geographique" dg ON dc."ClientID" = dg."ClientID"
LEFT  JOIN public."dim_offre" do_ ON dc."ClientID" = do_."ClientID"
ORDER BY fp."ClientFK", fp."DateFK"
"""


def load_monthly_performance(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(_MONTHLY_QUERY, engine)


def database_stats(engine: Engine) -> dict:
    """Lightweight counts used by the Monitoring page — no heavy joins."""
    with engine.connect() as conn:
        tables = [
            "dim_client",
            "dim_geographique",
            "dim_offre",
            "Fact_performance_client",
            "Fact_churn",
        ]
        counts = {}
        for t in tables:
            result = conn.exec_driver_sql(f'SELECT COUNT(*) FROM public."{t}"')
            counts[t] = result.scalar_one()
    return counts
