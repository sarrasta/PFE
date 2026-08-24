import { useNavigate } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { DashboardApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { InfoHint } from "../components/Tooltip";
import { formatDateTime, formatInt, formatPercent, formatTnd } from "../utils/format";
import { CHART_GRID, RISK_COLOR, SEGMENT_COLOR, tooltipItemStyle, tooltipLabelStyle, tooltipStyle } from "../chartTheme";
import { IconTarget, IconTrendDown, IconUsers, IconWallet } from "../components/Icons";

export function Dashboard() {
  const { data, loading, error, warmingUp, reload } = useApiResource(DashboardApi.get, []);
  const navigate = useNavigate();

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Tableau de bord</h2>
          <p>Vue d'ensemble du portefeuille de 10 000 clients — mise à jour {formatDateTime(data?.last_updated)}</p>
        </div>
      </div>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && <DashboardBody data={data} navigate={navigate} />}
      </AsyncSection>
    </div>
  );
}

function DashboardBody({ data, navigate }) {
  const k = data.kpis;
  const riskData = ["Faible", "Moyen", "Eleve"].map((label) => ({
    label,
    value: k.risk_distribution?.[label] || 0,
  }));
  const segmentData = Object.entries(k.segment_distribution || {}).map(([label, value]) => ({ label, value }));
  const urgency = data.retention_summary?.urgency_buckets || {};

  const arpuDelta = k.avg_arpu_predicted - k.avg_arpu_current;

  return (
    <>
      <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
        <KpiCard
          icon={<IconUsers size={18} />}
          label="Clients au portefeuille"
          value={formatInt(k.n_clients_total)}
        />
        <KpiCard
          icon={<IconTrendDown size={18} />}
          label="Clients à risque élevé"
          value={`${formatInt(k.high_risk_clients)} (${k.risk_distribution_pct?.Eleve ?? 0}%)`}
          delta={{ direction: "up", label: `${formatPercent(k.avg_propensity_score)} propension moy.` }}
          deltaTone="up-is-bad"
        />
        <KpiCard
          icon={<IconWallet size={18} />}
          label="ARPU moyen (actuel → prédit)"
          value={`${formatTnd(k.avg_arpu_current, 1)} → ${formatTnd(k.avg_arpu_predicted, 1)}`}
          delta={{ direction: arpuDelta >= 0 ? "up" : "down", label: `${arpuDelta >= 0 ? "+" : ""}${arpuDelta.toFixed(2)} TND` }}
        />
        <KpiCard
          icon={<IconTarget size={18} />}
          label="Gain de rétention potentiel"
          value={formatTnd(k.total_expected_gain_tnd)}
          delta={{ direction: "up", label: `${formatInt(k.n_roi_positive_clients)} clients ROI+` }}
        />
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card
          title="Répartition par niveau de risque"
          subtitle="Objectif 2 — clustering + classification supervisée"
        >
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={riskData}
                dataKey="value"
                nameKey="label"
                innerRadius={62}
                outerRadius={95}
                paddingAngle={3}
                stroke="var(--surface-card)"
                strokeWidth={2}
              >
                {riskData.map((entry) => (
                  <Cell key={entry.label} fill={RISK_COLOR[entry.label]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipLabelStyle}
                itemStyle={tooltipItemStyle}
                formatter={(value) => formatInt(value)}
              />
              <Legend
                verticalAlign="bottom"
                formatter={(value) => <span style={{ color: "var(--text-secondary)", fontSize: 12.5 }}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card
          title="Répartition par segment comportemental"
          subtitle="Objectif 4 — segmentation K-Means (scores composites)"
          actions={<button className="btn btn-ghost btn-sm" onClick={() => navigate("/segmentation")}>Détails</button>}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={segmentData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid horizontal={false} stroke={CHART_GRID} />
              <XAxis type="number" tick={{ fontSize: 11.5, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11.5, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatInt(v)} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={22}>
                {segmentData.map((entry) => (
                  <Cell key={entry.label} fill={SEGMENT_COLOR[entry.label] || "#2a78d6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="page-grid grid-3">
        <Card title="Revenu annuel à risque" subtitle="Σ ARPU × P(churn) × 12 mois">
          <div className="kpi-value" style={{ fontSize: 30 }}>{formatTnd(k.total_revenue_at_risk_tnd)}</div>
        </Card>
        <Card
          title={<span style={{ display: "flex", alignItems: "center", gap: 6 }}>Urgence de campagnes <InfoHint text="Rouge = risque élevé, Orange = risque moyen, Vert = risque faible (Objectif 5)" /></span>}
        >
          <UrgencyRow label="Immédiat (Rouge)" value={urgency.Rouge} tone="critical" />
          <UrgencyRow label="Préventif (Orange)" value={urgency.Orange} tone="warning" />
          <UrgencyRow label="Fidélisation (Vert)" value={urgency.Vert} tone="good" />
        </Card>
        <Card title="Base de données" subtitle="Entrepôt DW_TT — lignes restaurées">
          {Object.entries(k.database_row_counts || {}).map(([table, count]) => (
            <div key={table} style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, padding: "5px 0" }}>
              <span style={{ color: "var(--text-secondary)" }}>{table}</span>
              <strong>{formatInt(count)}</strong>
            </div>
          ))}
        </Card>
      </div>
    </>
  );
}

function UrgencyRow({ label, value, tone }) {
  const max = 10000;
  const pct = Math.min(100, ((value || 0) / max) * 100);
  const color = `var(--status-${tone})`;
  return (
    <div style={{ marginBottom: "var(--space-3)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 4 }}>
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <strong>{formatInt(value)}</strong>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}
