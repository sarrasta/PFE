import { useNavigate, useParams } from "react-router-dom";
import { ClientsApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { RiskBadge } from "../components/Badge";
import { InfoHint } from "../components/Tooltip";
import { IconChevronLeft } from "../components/Icons";
import { formatInt, formatPercent, formatTnd } from "../utils/format";

export function ClientDetail() {
  const { clientId } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, warmingUp, reload } = useApiResource(() => ClientsApi.get(clientId), [clientId]);
  const c = data?.client;

  return (
    <div>
      <button className="btn btn-ghost btn-sm" onClick={() => navigate(-1)} style={{ marginBottom: "var(--space-4)" }}>
        <IconChevronLeft size={15} /> Retour
      </button>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {c && (
          <>
            <div className="page-header">
              <div>
                <h2>Client #{c.client_id} <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>({c.client_ref})</span></h2>
                <p>{c.region} · {c.gouvernorat} · {c.type_abo} · Client depuis {formatInt(c.anciennete_mois)} mois</p>
              </div>
              <div className="page-actions">
                <RiskBadge level={c.risk_label} />
              </div>
            </div>

            <div className="page-grid grid-4" style={{ marginBottom: "var(--space-5)" }}>
              <ScoreCard label="P(Churn) — Obj. 1" hint="Classification SVC-RBF, calibrée sur 422 clients étiquetés" value={formatPercent(c.churn_proba)} />
              <ScoreCard label="Propension churn — Obj. 6" hint="Score continu Ridge + calibration isotonique" value={formatPercent(c.propensity_score)} />
              <ScoreCard label="Réponse à une offre — Obj. 8" hint="Probabilité d'acceptation d'une offre de rétention" value={formatPercent(c.response_proba)} />
              <ScoreCard label="Gain net attendu — Obj. 9" hint="P(churn) × P(réponse) × ARPU prédit × 12 mois − coût" value={formatTnd(c.gain_net, 1)} tone={c.gain_net > 0 ? "good" : "critical"} />
            </div>

            <div className="page-grid grid-2" style={{ marginBottom: "var(--space-5)" }}>
              <Card title="Profil de risque & segment">
                <DetailRow label="Niveau de risque (Obj. 2)" value={<RiskBadge level={c.risk_label} />} />
                <DetailRow label="Segment comportemental (Obj. 4)" value={c.segment} />
                <DetailRow label="Segment commercial" value={c.client_segment} />
                <DetailRow label="Type d'abonnement" value={c.type_abo} />
                <DetailRow label="Offre actuelle" value={`${c.offre_actuelle || "—"} (${formatTnd(c.prix_offre, 0)})`} />
                <DetailRow label="Engagement restant" value={`${formatInt(c.engagement_restant)} mois`} />
                <DetailRow label="ROI de ciblage" value={`${c.roi != null ? c.roi.toFixed(1) : "—"}×`} />
              </Card>

              <Card title="Revenu (Objectif 7 — ARPU futur)">
                <DetailRow label="ARPU actuel (moyenne observée)" value={formatTnd(c.avg_arpu, 2)} />
                <DetailRow label="ARPU futur prédit" value={formatTnd(c.arpu_futur_predit, 2)} />
                <DetailRow label="Facture actuelle" value={formatTnd(c.avg_montant_facture, 2)} />
                <DetailRow label="Facture future prédite" value={formatTnd(c.facture_futur_predit, 2)} />
                <DetailRow label="Hors-forfait moyen" value={formatTnd(c.avg_hors_forfait, 2)} />
                <DetailRow label="Impayés (cumulé)" value={formatInt(c.total_impayes)} />
              </Card>
            </div>

            <Card title="Indicateurs comportementaux bruts" subtitle="Moyennes mensuelles observées sur l'entrepôt DW_TT">
              <div className="page-grid grid-4">
                <DetailRow label="Minutes / mois" value={formatInt(c.avg_minutes)} />
                <DetailRow label="Data (GB) / mois" value={c.avg_data_gb?.toFixed(1)} />
                <DetailRow label="SMS / mois" value={formatInt(c.avg_sms)} />
                <DetailRow label="NPS moyen" value={c.avg_nps?.toFixed(1)} />
                <DetailRow label="QoS moyen" value={c.avg_qos?.toFixed(1)} />
                <DetailRow label="Taux de coupure réseau" value={formatPercent(c.avg_drop_rate / 100)} />
                <DetailRow label="Minutes d'indisponibilité" value={c.avg_outage_min?.toFixed(1)} />
                <DetailRow label="Tickets SAV" value={formatInt(c.total_tickets)} />
              </div>
            </Card>
          </>
        )}
      </AsyncSection>
    </div>
  );
}

function ScoreCard({ label, hint, value, tone }) {
  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <span className="kpi-label" style={{ display: "flex", alignItems: "center", gap: 5 }}>
          {label} {hint && <InfoHint text={hint} />}
        </span>
      </div>
      <div className="kpi-value" style={tone === "critical" ? { color: "var(--status-critical)" } : tone === "good" ? { color: "var(--status-good)" } : undefined}>
        {value}
      </div>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid var(--border-hairline)", fontSize: 13 }}>
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}
