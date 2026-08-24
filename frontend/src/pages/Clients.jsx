import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ClientsApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection } from "../components/States";
import { Card } from "../components/Card";
import { Pagination, Table } from "../components/Table";
import { RiskBadge } from "../components/Badge";
import { IconSearch } from "../components/Icons";
import { formatPercent, formatTnd } from "../utils/format";

const PAGE_SIZE = 25;

export function Clients() {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [riskLabel, setRiskLabel] = useState("");
  const [segment, setSegment] = useState("");
  const [region, setRegion] = useState("");
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("gain_net");
  const [sortDir, setSortDir] = useState("desc");

  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput), 350);
    return () => clearTimeout(t);
  }, [searchInput]);

  useEffect(() => setPage(1), [search, riskLabel, segment, region]);

  const { data, loading, error, warmingUp, reload } = useApiResource(
    () => ClientsApi.list({ search, risk_label: riskLabel, segment, region, sort_by: sortBy, sort_dir: sortDir, page, page_size: PAGE_SIZE }),
    [search, riskLabel, segment, region, sortBy, sortDir, page]
  );
  const { data: filterOptions } = useApiResource(ClientsApi.filters, []);

  function handleSort(key) {
    if (sortBy === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortBy(key);
      setSortDir("desc");
    }
  }

  const columns = [
    { key: "client_id", label: "ID", sortable: true },
    { key: "region", label: "Région" },
    { key: "segment", label: "Segment" },
    { key: "risk_label", label: "Risque", render: (r) => <RiskBadge level={r.risk_label} /> },
    { key: "churn_proba", label: "P(Churn)", align: "right", sortable: true, render: (r) => formatPercent(r.churn_proba) },
    { key: "propensity_score", label: "Propension", align: "right", sortable: true, render: (r) => formatPercent(r.propensity_score) },
    { key: "avg_arpu", label: "ARPU actuel", align: "right", sortable: true, render: (r) => formatTnd(r.avg_arpu, 1) },
    { key: "gain_net", label: "Gain attendu", align: "right", sortable: true, render: (r) => formatTnd(r.gain_net, 1) },
  ];

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Clients</h2>
          <p>Rechercher, filtrer et prioriser les 10 000 clients du portefeuille</p>
        </div>
      </div>

      <div className="filter-bar">
        <div className="field" style={{ minWidth: 260 }}>
          <label>Recherche</label>
          <div className="search-input-wrap">
            <IconSearch size={15} />
            <input
              className="input"
              placeholder="ID client, région, offre..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
          </div>
        </div>
        <div className="field">
          <label>Niveau de risque</label>
          <select className="select" value={riskLabel} onChange={(e) => setRiskLabel(e.target.value)}>
            <option value="">Tous</option>
            {filterOptions?.risk_labels?.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Segment</label>
          <select className="select" value={segment} onChange={(e) => setSegment(e.target.value)}>
            <option value="">Tous</option>
            {filterOptions?.segments?.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Région</label>
          <select className="select" value={region} onChange={(e) => setRegion(e.target.value)}>
            <option value="">Toutes</option>
            {filterOptions?.regions?.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        {(riskLabel || segment || region || searchInput) && (
          <button className="btn btn-ghost btn-sm" onClick={() => { setRiskLabel(""); setSegment(""); setRegion(""); setSearchInput(""); }}>
            Réinitialiser
          </button>
        )}
      </div>

      <Card noPad>
        <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload} isEmpty={data && data.items.length === 0}>
          {data && (
            <>
              <Table
                columns={columns}
                rows={data.items}
                sortBy={sortBy}
                sortDir={sortDir}
                onSort={handleSort}
                onRowClick={(row) => navigate(`/clients/${row.client_id}`)}
              />
              <Pagination page={data.page} totalPages={data.total_pages} total={data.total} pageSize={PAGE_SIZE} onPageChange={setPage} />
            </>
          )}
        </AsyncSection>
      </Card>
    </div>
  );
}
