import { IconActivity, IconAlertTriangle, IconDatabase, IconRefresh } from "./Icons";

export function LoadingState({ label = "Chargement des données..." }) {
  return (
    <div className="state-block">
      <div className="spinner" />
      <p>{label}</p>
    </div>
  );
}

export function WarmingUpState() {
  return (
    <div className="state-block">
      <IconActivity size={32} />
      <h4>Entraînement des modèles en cours</h4>
      <p>
        Le pipeline ML recalcule les 9 objectifs à partir de l'entrepôt de données
        (généralement quelques minutes au premier démarrage). Cette page se
        mettra à jour automatiquement.
      </p>
      <div className="spinner" />
    </div>
  );
}

export function KpiSkeletons({ count = 4 }) {
  return (
    <div className="page-grid grid-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="skeleton skeleton-kpi" />
      ))}
    </div>
  );
}

export function ChartSkeleton() {
  return <div className="skeleton skeleton-chart" />;
}

export function EmptyState({ title = "Aucune donnée", message, icon }) {
  return (
    <div className="state-block">
      {icon || <IconDatabase size={32} />}
      <h4>{title}</h4>
      {message && <p>{message}</p>}
    </div>
  );
}

export function ErrorState({ title = "Une erreur est survenue", message, onRetry }) {
  return (
    <div className="state-block state-error">
      <IconAlertTriangle size={32} />
      <h4>{title}</h4>
      {message && <p>{message}</p>}
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry}>
          <IconRefresh size={14} /> Réessayer
        </button>
      )}
    </div>
  );
}

/** Wraps async-loaded content with consistent loading/warming-up/error/empty
 * handling — pass the fields straight from `useApiResource`. */
export function AsyncSection({ loading, warmingUp, error, onRetry, isEmpty, emptyProps, children }) {
  if (warmingUp) return <WarmingUpState />;
  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} onRetry={onRetry} />;
  if (isEmpty) return <EmptyState {...emptyProps} />;
  return children;
}
