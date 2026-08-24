import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SegmentationApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { IconLayers } from "../components/Icons";
import { formatInt, formatPercent, formatTnd } from "../utils/format";
import { CHART_GRID, SEGMENT_COLOR, tooltipItemStyle, tooltipLabelStyle, tooltipStyle } from "../chartTheme";

export function Segmentation() {
  const { data, loading, error, warmingUp, reload } = useApiResource(SegmentationApi.overview, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Segmentation comportementale</h2>
          <p>Objectif 4 — K-Means sur 4 scores composites (usage, prix, réseau, inactivité)</p>
        </div>
      </div>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && <SegmentationBody data={data} />}
      </AsyncSection>
    </div>
  );
}

function SegmentationBody({ data }) {
  const m = data.metrics || {};
  const countData = data.segment_names.map((name) => ({ label: name, value: data.segment_counts[name] || 0 }));
  const arpuData = data.segment_profile?.map((r) => ({ label: r.segment, value: r.avg_arpu })) || [];
  // Propensity (Obj. 6), not the raw churn_proba classifier (Obj. 1) — the
  // latter is intentionally near-saturated (SVC trained on 93.6% churners)
  // and would render as four indistinguishable ~90-100% bars here.
  const propensityData = data.segment_profile?.map((r) => ({ label: r.segment, value: r.propensity_score })) || [];

  return (
    <>
      <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
        <KpiCard icon={<IconLayers size={18} />} label="Score de silhouette" value={(m.silhouette ?? 0).toFixed(4)} />
        <KpiCard icon={<IconLayers size={18} />} label="Davies-Bouldin (↓ mieux)" value={(m.davies_bouldin ?? 0).toFixed(4)} />
        <KpiCard icon={<IconLayers size={18} />} label="Calinski-Harabasz" value={formatInt(m.calinski_harabasz)} />
        <KpiCard icon={<IconLayers size={18} />} label="Précision classificateur RF" value={formatPercent(m.classifier_cv_accuracy)} />
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card title="Répartition des segments" subtitle={`${data.segment_names.length} groupes — ${formatInt(countData.reduce((s, d) => s + d.value, 0))} clients`}>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={countData} dataKey="value" nameKey="label" innerRadius={60} outerRadius={95} paddingAngle={3} stroke="var(--surface-card)" strokeWidth={2}>
                {countData.map((entry) => <Cell key={entry.label} fill={SEGMENT_COLOR[entry.label] || "#2a78d6"} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatInt(v)} />
              <Legend verticalAlign="bottom" formatter={(v) => <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="ARPU moyen par segment" subtitle="TND / mois">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={arpuData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid horizontal={false} stroke={CHART_GRID} />
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11.5, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatTnd(v, 1)} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={22}>
                {arpuData.map((entry) => <Cell key={entry.label} fill={SEGMENT_COLOR[entry.label] || "#2a78d6"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
        <Card title="Propension au churn par segment" subtitle="Objectif 6 — score continu calibré">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={propensityData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid horizontal={false} stroke={CHART_GRID} />
              <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11.5, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => formatPercent(v)} />
              <Bar dataKey="value" fill="var(--status-critical)" radius={[0, 4, 4, 0]} barSize={22} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Profil détaillé des segments" noPad>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Segment</th>
                  <th className="num">Minutes</th>
                  <th className="num">Data (GB)</th>
                  <th className="num">ARPU</th>
                  <th className="num">Drop rate</th>
                  <th className="num">Ancienneté</th>
                </tr>
              </thead>
              <tbody>
                {data.segment_profile?.map((r) => (
                  <tr key={r.segment}>
                    <td>{r.segment}</td>
                    <td className="num">{formatInt(r.avg_minutes)}</td>
                    <td className="num">{r.avg_data_gb?.toFixed(1)}</td>
                    <td className="num">{formatTnd(r.avg_arpu, 1)}</td>
                    <td className="num">{r.avg_drop_rate?.toFixed(2)}%</td>
                    <td className="num">{formatInt(r.anciennete_mois)} mois</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}
