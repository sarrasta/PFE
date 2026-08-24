import { useState } from "react";
import { AdminApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { KpiCard } from "../components/KpiCard";
import { Badge } from "../components/Badge";
import { IconActivity, IconDatabase, IconRefresh } from "../components/Icons";
import { formatDateTime, formatInt } from "../utils/format";

const STATUS_TONE = { ready: "good", loading: "warning", error: "critical", not_initialized: "neutral" };
const STATUS_LABEL = { ready: "Opérationnel", loading: "Entraînement en cours", error: "Erreur", not_initialized: "Non initialisé" };

export function Monitoring() {
  const { data, loading, error, warmingUp, reload } = useApiResource(AdminApi.monitoring, []);
  const [refreshing, setRefreshing] = useState(false);
  const [message, setMessage] = useState(null);

  async function handleRefresh() {
    setRefreshing(true);
    setMessage(null);
    try {
      await AdminApi.refresh();
      setMessage("Actualisation du pipeline ML démarrée — cela peut prendre plusieurs minutes.");
      setTimeout(reload, 3000);
    } catch (err) {
      setMessage(err.message || "Échec du déclenchement de l'actualisation.");
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Monitoring</h2>
          <p>État du service, du pipeline ML et de l'entrepôt de données</p>
        </div>
        <div className="page-actions">
          <button className="btn btn-primary" onClick={handleRefresh} disabled={refreshing}>
            <IconRefresh size={15} /> {refreshing ? "Démarrage..." : "Réentraîner les modèles"}
          </button>
        </div>
      </div>

      {message && <div className="login-error" style={{ marginBottom: "var(--space-5)", maxWidth: 520 }}>{message}</div>}

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && (
          <>
            <div className="page-grid grid-4" style={{ marginBottom: "var(--space-6)" }}>
              <KpiCard icon={<IconActivity size={18} />} label="Disponibilité du service" value={formatUptime(data.service_uptime_seconds)} />
              <KpiCard icon={<IconDatabase size={18} />} label="Clients scorés" value={formatInt(data.ml_pipeline.clients_scored)} />
              <KpiCard icon={<IconActivity size={18} />} label="Durée du dernier entraînement" value={data.ml_pipeline.last_duration_seconds ? `${Math.round(data.ml_pipeline.last_duration_seconds)}s` : "—"} />
              <div className="kpi-card">
                <div className="kpi-top">
                  <span className="kpi-label">État du pipeline ML</span>
                </div>
                <Badge tone={STATUS_TONE[data.ml_pipeline.status] || "neutral"}>{STATUS_LABEL[data.ml_pipeline.status] || data.ml_pipeline.status}</Badge>
              </div>
            </div>

            {data.ml_pipeline.error_message && (
              <Card title="Dernière erreur du pipeline" style={{ marginBottom: "var(--space-6)" }}>
                <pre className="metrics-raw">{data.ml_pipeline.error_message}</pre>
              </Card>
            )}

            <div className="page-grid grid-2">
              <Card title="Dernière actualisation">
                <p style={{ fontSize: 13.5 }}>{formatDateTime(data.ml_pipeline.last_updated)}</p>
              </Card>
              <Card title="Lignes restaurées — entrepôt DW_TT" noPad>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead><tr><th>Table</th><th className="num">Lignes</th></tr></thead>
                    <tbody>
                      {Object.entries(data.database.row_counts || {}).map(([t, c]) => (
                        <tr key={t}><td>{t}</td><td className="num">{formatInt(c)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </>
        )}
      </AsyncSection>
    </div>
  );
}

function formatUptime(seconds) {
  if (!seconds) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}min`;
  return `${m}min`;
}
