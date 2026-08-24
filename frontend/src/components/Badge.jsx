const RISK_TONE = { Faible: "good", Moyen: "warning", Eleve: "critical" };

export function Badge({ tone = "neutral", children }) {
  return (
    <span className={`badge badge-${tone}`}>
      <span className="dot" />
      {children}
    </span>
  );
}

/** Risk-level badge (Faible/Moyen/Eleve) — status color, always icon+label. */
export function RiskBadge({ level }) {
  return <Badge tone={RISK_TONE[level] || "neutral"}>{level || "Inconnu"}</Badge>;
}
