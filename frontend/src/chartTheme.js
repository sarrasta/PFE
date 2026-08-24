/* Mirrors the CSS custom properties in styles/tokens.css — duplicated here
 * as plain JS because Recharts (SVG) needs literal color strings, not
 * `var(--...)`, for fills/strokes. Keep in sync with tokens.css. */
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];

export const STATUS = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
};

export const RISK_COLOR = {
  Faible: STATUS.good,
  Moyen: STATUS.warning,
  Eleve: STATUS.critical,
};

export const SEGMENT_COLOR = {
  "Fort Usage": SERIES[0],
  "Sensibles au Prix": SERIES[1],
  "Risque Reseau": SERIES[2],
  "Clients Inactifs": SERIES[3],
};

export const SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"];

export const CHART_GRID = "#e1e0d9";
export const CHART_AXIS = "#c3c2b7";
export const TEXT_MUTED = "#8b93a1";
export const TEXT_SECONDARY = "#565d6b";

export const tooltipStyle = {
  background: "#14171c",
  border: "none",
  borderRadius: 8,
  color: "#fff",
  fontSize: 12.5,
  padding: "8px 12px",
  boxShadow: "0 12px 32px rgba(20,23,28,0.25)",
};
export const tooltipLabelStyle = { color: "#fff", fontWeight: 600, marginBottom: 4 };
export const tooltipItemStyle = { color: "#fff" };
