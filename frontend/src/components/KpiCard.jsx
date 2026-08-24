import { IconArrowDown, IconArrowUp } from "./Icons";

/**
 * `delta` is optional: { direction: "up" | "down" | "flat", label: string }.
 * `deltaTone` lets the caller say whether "up" is good or bad for this
 * particular metric (e.g. churn going up is bad, revenue going up is good).
 */
export function KpiCard({ icon, label, value, delta, deltaTone = "up-is-good" }) {
  let deltaClass = "flat";
  if (delta?.direction === "up") deltaClass = deltaTone === "up-is-good" ? "up" : "down";
  if (delta?.direction === "down") deltaClass = deltaTone === "up-is-good" ? "down" : "up";

  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <span className="kpi-label">{label}</span>
        {icon && <div className="kpi-icon">{icon}</div>}
      </div>
      <div className="kpi-value">{value}</div>
      {delta && (
        <span className={`kpi-delta ${deltaClass}`}>
          {delta.direction === "up" && <IconArrowUp size={12} />}
          {delta.direction === "down" && <IconArrowDown size={12} />}
          {delta.label}
        </span>
      )}
    </div>
  );
}
