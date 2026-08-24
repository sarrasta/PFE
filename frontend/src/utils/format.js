export function formatNumber(value, opts = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("fr-FR", opts).format(value);
}

export function formatInt(value) {
  return formatNumber(value, { maximumFractionDigits: 0 });
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatNumber(value * 100, { maximumFractionDigits: digits, minimumFractionDigits: digits })}%`;
}

export function formatTnd(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${formatNumber(value, { maximumFractionDigits: digits, minimumFractionDigits: digits })} TND`;
}

export function formatDateTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("fr-FR", { dateStyle: "medium", timeStyle: "short" });
}
