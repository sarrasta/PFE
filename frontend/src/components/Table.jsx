import { IconChevronDown } from "./Icons";
import { EmptyState } from "./States";

/**
 * Presentational table. `columns`: [{ key, label, align, sortable, render }]
 * Sorting/pagination state is owned by the caller (server-side for large
 * datasets like the Clients page).
 */
export function Table({ columns, rows, sortBy, sortDir, onSort, onRowClick, emptyMessage }) {
  if (!rows || rows.length === 0) {
    return <EmptyState title="Aucun résultat" message={emptyMessage || "Ajustez vos filtres ou votre recherche."} />;
  }

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`${col.align === "right" ? "num" : ""} ${col.sortable ? "sortable" : ""}`}
                onClick={col.sortable ? () => onSort?.(col.key) : undefined}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  {col.label}
                  {col.sortable && sortBy === col.key && (
                    <IconChevronDown size={12} style={{ transform: sortDir === "asc" ? "rotate(180deg)" : "none" }} />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.id ?? row.client_id ?? i} className={onRowClick ? "clickable" : ""} onClick={() => onRowClick?.(row)}>
              {columns.map((col) => (
                <td key={col.key} className={col.align === "right" ? "num" : ""}>
                  {col.render ? col.render(row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Pagination({ page, totalPages, total, pageSize, onPageChange }) {
  if (total === 0) return null;
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(total, page * pageSize);
  return (
    <div className="pagination">
      <span>
        {start.toLocaleString("fr-FR")}–{end.toLocaleString("fr-FR")} sur {total.toLocaleString("fr-FR")}
      </span>
      <div className="pagination-controls">
        <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          Précédent
        </button>
        <span>Page {page} / {totalPages}</span>
        <button className="btn btn-ghost btn-sm" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
          Suivant
        </button>
      </div>
    </div>
  );
}
