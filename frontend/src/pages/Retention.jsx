import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RetentionApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { Table } from "../components/Table";
import { RiskBadge } from "../components/Badge";
import { InfoHint } from "../components/Tooltip";
import { IconRefresh, IconTarget, IconWallet } from "../components/Icons";
import { formatInt, formatPercent, formatTnd } from "../utils/format";
import { CHART_GRID, STATUS, tooltipItemStyle, tooltipLabelStyle, tooltipStyle } from "../chartTheme";

export function Retention() {
  const navigate = useNavigate();
  const matrix = useApiResource(RetentionApi.targetingMatrix, []);
  const gain = useApiResource(RetentionApi.gainOverview, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Campagnes de rétention</h2>
          <p>Objectifs 5 (ciblage), 8 (réponse à l'offre) et 9 (gain net attendu)</p>
        </div>
      </div>

      <AsyncSection loading={matrix.loading || gain.loading} warmingUp={matrix.warmingUp || gain.warmingUp} error={matrix.error || gain.error} onRetry={() => { matrix.reload(); gain.reload(); }}>
        {matrix.data && gain.data && <RetentionBody matrix={matrix.data} gain={gain.data} navigate={navigate} />}
      </AsyncSection>
    </div>
  );
}

function RetentionBody({ matrix, gain, navigate }) {
  const results = gain.metrics?.results || {};
  const lift = gain.metrics?.lift_at_pct || {};
  const liftData = Object.entries(lift).map(([k, v]) => ({ label: k.replace("top_", "Top ").replace("pct", "%"), value: v }));

  const matrixColumns = [
    { key: "priority", label: "#", render: (r) => `P${r.priority}` },
    { key: "segment", label: "Segment" },
    { key: "risk_label", label: "Risque", render: (r) => <RiskBadge level={r.risk_label} /> },
    { key: "n_clients", label: "Clients", align: "right", render: (r) => formatInt(r.n_clients) },
    { key: "churn_proba_mean", label: "P(churn)", align: "right", render: (r) => formatPercent(r.churn_proba_mean) },
    { key: "arpu_mean", label: "ARPU", align: "right", render: (r) => formatTnd(r.arpu_mean, 1) },
    { key: "roi_net_tnd", label: "ROI net", align: "right", render: (r) => formatTnd(r.roi_net_tnd) },
    { key: "campaign", label: "Campagne recommandée" },
  ];

  const topGainColumns = [
    { key: "client_id", label: "ID" },
    { key: "segment", label: "Segment" },
    { key: "risk_label", label: "Risque", render: (r) => <RiskBadge level={r.risk_label} /> },
    { key: "propensity_score", label: "Propension", align: "right", render: (r) => formatPercent(r.propensity_score) },
    { key: "response_proba", label: "P(réponse)", align: "right", render: (r) => formatPercent(r.response_proba) },
    { key: "gain_net", label: "Gain net", align: "right", render: (r) => formatTnd(r.gain_net, 1) },
  ];

  return (
    <>
      <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
        <KpiCard icon={<IconWallet size={18} />} label="Gain net total potentiel" value={formatTnd(results.gain_total_tnd)} />
        <KpiCard icon={<IconTarget size={18} />} label="Clients ROI positif" value={`${formatInt(results.n_roi_positive)} (${formatPercent(results.pct_roi_positive)})`} />
        <KpiCard icon={<IconWallet size={18} />} label="ROI moyen" value={`${(results.roi_mean ?? 0).toFixed(1)}×`} />
        <KpiCard icon={<IconTarget size={18} />} label="ROI net — matrice de ciblage" value={formatTnd(matrix.total_roi_net_tnd)} />
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card title="Lift du ciblage par gain attendu" subtitle="vs. ciblage aléatoire (Objectif 9)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={liftData}>
              <CartesianGrid vertical={false} stroke={CHART_GRID} />
              <XAxis dataKey="label" tick={{ fontSize: 11.5, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}×`} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => [`${v}×`, "lift"]} />
              <Bar dataKey="value" fill={STATUS.good} radius={[3, 3, 0, 0]} barSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <ScenarioSimulator initialMetrics={gain.metrics} />
      </div>

      <Card
        title="Matrice de ciblage — 12 micro-profils"
        subtitle={<span>Segment × Niveau de risque · Rétention assumée {formatPercent(matrix.retention_rate_assumed)} · Coût offre {formatTnd(matrix.offer_cost_tnd)}</span>}
        noPad
      >
        <Table columns={matrixColumns} rows={matrix.micro_profiles} />
      </Card>

      <div style={{ height: "var(--space-6)" }} />

      <Card title="Top 20 clients par gain net attendu" noPad>
        <Table columns={topGainColumns} rows={gain.top_clients_by_gain} onRowClick={(row) => navigate(`/clients/${row.client_id}`)} />
      </Card>
    </>
  );
}

function ScenarioSimulator({ initialMetrics }) {
  const [ltv, setLtv] = useState(initialMetrics?.parameters?.ltv_horizon_months ?? 12);
  const [cost, setCost] = useState(initialMetrics?.parameters?.offer_cost_tnd ?? 15);
  const [result, setResult] = useState(initialMetrics);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      const res = await RetentionApi.simulateScenario(Number(ltv), Number(cost));
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card
      title={<span style={{ display: "flex", alignItems: "center", gap: 6 }}>Simulateur de scénario <InfoHint text="Recalcule le gain net (Objectif 9) sans ré-entraîner les modèles" /></span>}
      subtitle="Ajustez l'horizon de rétention et le coût de l'offre"
    >
      <div style={{ display: "flex", gap: "var(--space-4)", marginBottom: "var(--space-4)" }}>
        <div className="field">
          <label>Horizon LTV (mois)</label>
          <input className="input" type="number" min={1} max={60} value={ltv} onChange={(e) => setLtv(e.target.value)} />
        </div>
        <div className="field">
          <label>Coût de l'offre (TND)</label>
          <input className="input" type="number" min={1} max={1000} value={cost} onChange={(e) => setCost(e.target.value)} />
        </div>
        <div className="field" style={{ justifyContent: "flex-end" }}>
          <button className="btn btn-primary" onClick={run} disabled={loading}>
            <IconRefresh size={14} /> {loading ? "Calcul..." : "Recalculer"}
          </button>
        </div>
      </div>

      {result && (
        <div className="page-grid grid-3">
          <SimStat label="Gain total" value={formatTnd(result.results?.gain_total_tnd)} />
          <SimStat label="Clients ROI+" value={formatInt(result.results?.n_roi_positive)} />
          <SimStat label="ROI moyen" value={`${(result.results?.roi_mean ?? 0).toFixed(1)}×`} />
        </div>
      )}
    </Card>
  );
}

function SimStat({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
