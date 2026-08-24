import { useState } from "react";
import { ModelsApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { IconBarChart, IconChevronDown } from "../components/Icons";
import { formatDateTime, formatInt, formatPercent, formatTnd } from "../utils/format";

export function ModelPerformance() {
  const { data, loading, error, warmingUp, reload } = useApiResource(ModelsApi.list, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Performance des modèles</h2>
          <p>Les 9 objectifs ML, avec les métriques de validation croisée recalculées à chaque entraînement du pipeline</p>
        </div>
      </div>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && (
          <>
            <p style={{ fontSize: 12.5, color: "var(--text-muted)", marginBottom: "var(--space-5)" }}>
              Dernier entraînement : {formatDateTime(data.last_updated)}
              {data.last_duration_seconds ? ` · durée ${Math.round(data.last_duration_seconds)}s` : ""}
            </p>
            <div className="page-grid grid-2">
              {data.models.map((m) => <ObjectiveCard key={m.id} entry={m} />)}
            </div>
          </>
        )}
      </AsyncSection>
    </div>
  );
}

function ObjectiveCard({ entry }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <Card
      title={<span style={{ display: "flex", alignItems: "center", gap: 8 }}><IconBarChart size={16} /> {entry.name}</span>}
      subtitle={entry.type}
    >
      <ObjectiveSummary entry={entry} />
      <button className="btn btn-ghost btn-sm" style={{ marginTop: "var(--space-3)" }} onClick={() => setExpanded((e) => !e)}>
        <IconChevronDown size={13} style={{ transform: expanded ? "rotate(180deg)" : "none" }} />
        {expanded ? "Masquer le détail" : "Voir le détail complet"}
      </button>
      {expanded && (
        <pre className="metrics-raw">{JSON.stringify(entry.metrics, null, 2)}</pre>
      )}
    </Card>
  );
}

function Row({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", fontSize: 13 }}>
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ObjectiveSummary({ entry }) {
  const m = entry.metrics || {};
  switch (entry.id) {
    case "objective_1_churn":
      return (
        <>
          <Row label="Modèle" value={m.model} />
          <Row label="AUC-ROC (CV 5-fold)" value={(m.cv_auc_mean ?? 0).toFixed(4)} />
          <Row label="Écart-type" value={`± ${(m.cv_auc_std ?? 0).toFixed(4)}`} />
        </>
      );
    case "objective_2_risk":
      return (
        <>
          <Row label="Silhouette (k=3)" value={(m.silhouette_k3 ?? 0).toFixed(4)} />
          <Row label="Précision RF (CV)" value={formatPercent(m.rf_cv_accuracy)} />
          <Row label="F1-macro RF (CV)" value={formatPercent(m.rf_cv_f1_macro)} />
        </>
      );
    case "objective_3_business_metrics":
      return (
        <>
          <Row label="Meilleur modèle" value={m.best_model} />
          <Row label="Échantillon" value={`${formatInt(m.n_samples)} clients (prévalence ${formatPercent(m.prevalence)})`} />
          <Row label="Valeur nette optimale" value={formatTnd(m.cost_benefit?.net_value_tnd)} />
          <Row label="Seuil optimal" value={m.cost_benefit?.optimal_threshold} />
        </>
      );
    case "objective_4_segmentation":
      return (
        <>
          <Row label="Silhouette" value={(m.silhouette ?? 0).toFixed(4)} />
          <Row label="Davies-Bouldin" value={(m.davies_bouldin ?? 0).toFixed(4)} />
          <Row label="Précision classificateur" value={formatPercent(m.classifier_cv_accuracy)} />
        </>
      );
    case "objective_5_retention":
      return (
        <>
          <Row label="Micro-profils" value={m.n_micro_profiles} />
          <Row label="ROI net total" value={formatTnd(m.total_roi_net_tnd)} />
        </>
      );
    case "objective_6_propensity":
      return (
        <>
          <Row label="Meilleur modèle" value={m.best_model} />
          <Row label="AUC (meilleur)" value={m.models?.[m.best_model]?.auc?.toFixed(4)} />
          <Row label="Score moyen déployé" value={formatPercent(m.deployment?.score_mean)} />
        </>
      );
    case "objective_7_arpu":
      return (
        <>
          <Row label="Modèle (ARPU)" value={m.best_model_arpu} />
          <Row label="R² (ARPU)" value={m.models?.[m.best_model_arpu]?.arpu?.r2?.toFixed(4)} />
          <Row label="MAE (ARPU)" value={formatTnd(m.models?.[m.best_model_arpu]?.arpu?.mae, 2)} />
          <Row label="Features" value={m.n_features} />
        </>
      );
    case "objective_8_response":
      return (
        <>
          <Row label="Meilleur modèle" value={m.best_model} />
          <Row label="AUC (meilleur)" value={m.models?.[m.best_model]?.auc?.toFixed(4)} />
          <Row label="Taux de réponse synthétique" value={formatPercent(m.deployment?.synthetic_response_rate)} />
        </>
      );
    case "objective_9_gain":
      return (
        <>
          <Row label="Clients ROI+" value={formatInt(m.results?.n_roi_positive)} />
          <Row label="Gain total" value={formatTnd(m.results?.gain_total_tnd)} />
          <Row label="Lift Top 10%" value={`${m.lift_at_pct?.top_10pct ?? "—"}×`} />
        </>
      );
    default:
      return <p style={{ fontSize: 12.5, color: "var(--text-muted)" }}>Voir le détail complet ci-dessous.</p>;
  }
}
