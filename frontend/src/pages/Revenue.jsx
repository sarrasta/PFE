import { useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RevenueApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { Table } from "../components/Table";
import { IconArrowDown, IconArrowUp, IconWallet } from "../components/Icons";
import { formatTnd } from "../utils/format";
import { CHART_GRID, SERIES, tooltipItemStyle, tooltipLabelStyle, tooltipStyle } from "../chartTheme";

export function Revenue() {
  const navigate = useNavigate();
  const { data, loading, error, warmingUp, reload } = useApiResource(RevenueApi.overview, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Revenus & ARPU</h2>
          <p>Objectif 7 — prédiction de l'ARPU et de la facture futurs (régression sur split temporel)</p>
        </div>
      </div>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && <RevenueBody data={data} navigate={navigate} />}
      </AsyncSection>
    </div>
  );
}

function RevenueBody({ data, navigate }) {
  const m = data.metrics || {};
  const bestArpu = m.models?.[m.best_model_arpu]?.arpu || {};

  const growthColumns = [
    { key: "client_id", label: "ID" },
    { key: "region", label: "Région" },
    { key: "segment", label: "Segment" },
    { key: "avg_arpu", label: "ARPU actuel", align: "right", render: (r) => formatTnd(r.avg_arpu, 1) },
    { key: "arpu_futur_predit", label: "ARPU prédit", align: "right", render: (r) => formatTnd(r.arpu_futur_predit, 1) },
    { key: "arpu_delta", label: "Δ", align: "right", render: (r) => <span style={{ color: r.arpu_delta >= 0 ? "var(--status-good)" : "var(--status-critical)" }}>{r.arpu_delta >= 0 ? "+" : ""}{r.arpu_delta.toFixed(1)} TND</span> },
  ];

  return (
    <>
      <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
        <KpiCard icon={<IconWallet size={18} />} label="Modèle retenu (ARPU)" value={m.best_model_arpu || "—"} />
        <KpiCard icon={<IconWallet size={18} />} label="R² (ARPU futur)" value={(bestArpu.r2 ?? 0).toFixed(4)} />
        <KpiCard icon={<IconWallet size={18} />} label="MAE" value={formatTnd(bestArpu.mae, 2)} />
        <KpiCard icon={<IconWallet size={18} />} label="ARPU moyen (actuel → prédit)" value={`${formatTnd(data.arpu_current_mean, 1)} → ${formatTnd(data.arpu_predicted_mean, 1)}`} />
      </div>

      <Card title="ARPU actuel vs. prédit par segment" subtitle="TND / mois" style={{ marginBottom: "var(--space-6)" }}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.arpu_by_segment}>
            <CartesianGrid vertical={false} stroke={CHART_GRID} />
            <XAxis dataKey="segment" tick={{ fontSize: 11, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatTnd(v, 1)} />
            <Legend formatter={(v) => <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>{v}</span>} />
            <Bar dataKey="avg_arpu" name="ARPU actuel" fill={SERIES[0]} radius={[4, 4, 0, 0]} barSize={26} />
            <Bar dataKey="arpu_futur_predit" name="ARPU prédit" fill={SERIES[1]} radius={[4, 4, 0, 0]} barSize={26} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <div className="page-grid grid-2">
        <Card title={<span style={{ display: "flex", alignItems: "center", gap: 6 }}><IconArrowUp size={15} style={{ color: "var(--status-good)" }} /> Plus forte croissance ARPU prédite</span>} noPad>
          <Table columns={growthColumns} rows={data.top_arpu_growth} onRowClick={(row) => navigate(`/clients/${row.client_id}`)} />
        </Card>
        <Card title={<span style={{ display: "flex", alignItems: "center", gap: 6 }}><IconArrowDown size={15} style={{ color: "var(--status-critical)" }} /> Plus forte baisse ARPU prédite</span>} noPad>
          <Table columns={growthColumns} rows={data.top_arpu_decline} onRowClick={(row) => navigate(`/clients/${row.client_id}`)} />
        </Card>
      </div>
    </>
  );
}
