import { useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChurnApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { Table } from "../components/Table";
import { RiskBadge } from "../components/Badge";
import { IconActivity, IconTarget, IconTrendDown } from "../components/Icons";
import { formatInt, formatPercent, formatTnd } from "../utils/format";
import { CHART_GRID, SEGMENT_COLOR, STATUS, tooltipItemStyle, tooltipLabelStyle, tooltipStyle } from "../chartTheme";

export function ChurnRisk() {
  const navigate = useNavigate();
  const overview = useApiResource(ChurnApi.overview, []);
  const topRisk = useApiResource(() => ChurnApi.topRisk(15), []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Churn & Risque</h2>
          <p>Objectifs 1 (classification), 2 (niveau de risque) et 6 (score de propension continu)</p>
        </div>
      </div>

      <AsyncSection loading={overview.loading} warmingUp={overview.warmingUp} error={overview.error} onRetry={overview.reload}>
        {overview.data && <ChurnBody data={overview.data} topRisk={topRisk} navigate={navigate} />}
      </AsyncSection>
    </div>
  );
}

function ChurnBody({ data, topRisk, navigate }) {
  const churnHist = (data.churn_proba_histogram || []).map((b) => ({ x: `${Math.round(b.bucket_start * 100)}%`, count: b.count }));
  const propHist = (data.propensity_histogram || []).map((b) => ({ x: `${Math.round(b.bucket_start * 100)}%`, count: b.count }));
  const bySegment = (data.propensity_by_segment || []).map((r) => ({ label: r.segment, value: r.propensity_score }));

  const topRiskColumns = [
    { key: "client_id", label: "ID" },
    { key: "region", label: "Région" },
    { key: "segment", label: "Segment" },
    { key: "risk_label", label: "Risque", render: (r) => <RiskBadge level={r.risk_label} /> },
    { key: "propensity_score", label: "Propension", align: "right", render: (r) => formatPercent(r.propensity_score) },
    { key: "avg_arpu", label: "ARPU", align: "right", render: (r) => formatTnd(r.avg_arpu, 1) },
    { key: "gain_net", label: "Gain attendu", align: "right", render: (r) => formatTnd(r.gain_net, 1) },
  ];

  return (
    <>
      <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
        <KpiCard icon={<IconTrendDown size={18} />} label="Obj. 1 — AUC-ROC (churn)" value={(data.metrics?.cv_auc_mean ?? 0).toFixed(4)} />
        <KpiCard icon={<IconActivity size={18} />} label="Obj. 2 — Précision RF (risque)" value={formatPercent(data.risk_metrics?.rf_cv_accuracy)} />
        <KpiCard icon={<IconTarget size={18} />} label="Obj. 6 — Meilleur modèle" value={data.propensity_metrics?.best_model || "—"} />
        <KpiCard icon={<IconTrendDown size={18} />} label="Clients à risque élevé" value={formatInt(data.high_risk_count)} />
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card title="Distribution — probabilité de churn (Obj. 1)" subtitle="Classification SVC-RBF">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={churnHist}>
              <CartesianGrid vertical={false} stroke={CHART_GRID} />
              <XAxis dataKey="x" tick={{ fontSize: 10.5, fill: "var(--text-muted)" }} interval={2} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => [formatInt(v), "clients"]} />
              <Bar dataKey="count" fill={STATUS.critical} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Distribution — score de propension (Obj. 6)" subtitle="Régression Ridge calibrée">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={propHist}>
              <CartesianGrid vertical={false} stroke={CHART_GRID} />
              <XAxis dataKey="x" tick={{ fontSize: 10.5, fill: "var(--text-muted)" }} interval={2} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => [formatInt(v), "clients"]} />
              <Bar dataKey="count" fill={STATUS.warning} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card title="Propension moyenne par segment" subtitle="Croisement Objectif 4 × Objectif 6">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={bySegment} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid horizontal={false} stroke={CHART_GRID} />
              <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11.5, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatPercent(v)} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                {bySegment.map((entry) => <Cell key={entry.label} fill={SEGMENT_COLOR[entry.label] || "#2a78d6"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Profil par niveau de risque">
          {(data.profile_by_risk_level || []).map((row) => (
            <div key={row.risk_label} style={{ marginBottom: "var(--space-3)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <RiskBadge level={row.risk_label} />
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                  ARPU {formatTnd(row.avg_arpu, 1)} · Propension {formatPercent(row.propensity_score)}
                </span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${row.churn_proba * 100}%`, background: "var(--tt-red-600)" }} />
              </div>
            </div>
          ))}
        </Card>
      </div>

      <Card title="Clients prioritaires (propension la plus élevée)" noPad>
        <AsyncSection loading={topRisk.loading} warmingUp={topRisk.warmingUp} error={topRisk.error} onRetry={topRisk.reload} isEmpty={topRisk.data && topRisk.data.items.length === 0}>
          {topRisk.data && (
            <Table columns={topRiskColumns} rows={topRisk.data.items} onRowClick={(row) => navigate(`/clients/${row.client_id}`)} />
          )}
        </AsyncSection>
      </Card>
    </>
  );
}
