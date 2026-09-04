import { useMemo, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { IconExternalLink, IconPresentation } from "../components/Icons";

const REPORTS = [
  { id: "marketing", scope: "powerbi_marketing", label: "Direction Marketing", shortLabel: "Marketing", description: "Performance commerciale, campagnes et connaissance client", url: "https://app.powerbi.com/groups/me/reports/309d7ce1-55ab-4f49-ad9b-3e6424012a0c/9d553b1682b21a45140b?experience=power-bi" },
  { id: "reseau", scope: "powerbi_reseau", label: "Direction Réseau", shortLabel: "Réseau", description: "Qualité de service, couverture et performance du réseau", url: "https://app.powerbi.com/groups/me/reports/309d7ce1-55ab-4f49-ad9b-3e6424012a0c/046d86d6893e6b6bda68?experience=power-bi" },
  { id: "financiere", scope: "powerbi_financiere", label: "Direction Financière", shortLabel: "Financière", description: "Indicateurs financiers, revenus et suivi budgétaire", url: "https://app.powerbi.com/groups/me/reports/309d7ce1-55ab-4f49-ad9b-3e6424012a0c/30bcb442d97b030b9050?experience=power-bi" },
  { id: "regionale", scope: "powerbi_regionale", label: "Direction Régionale", shortLabel: "Régionale", description: "Pilotage territorial et suivi des performances régionales", url: "https://app.powerbi.com/groups/me/reports/309d7ce1-55ab-4f49-ad9b-3e6424012a0c/e6ab597709c74b039bd9?experience=power-bi" },
];

export function DirectionDashboards() {
  const { hasScope, user } = useAuth();
  const visibleReports = useMemo(() => REPORTS.filter((report) => hasScope(report.scope)), [hasScope]);
  const [selectedId, setSelectedId] = useState(() => visibleReports[0]?.id);
  const selected = visibleReports.find((report) => report.id === selectedId) || visibleReports[0];
  if (!selected) return null;

  return (
    <section className="powerbi-page">
      <div className="powerbi-hero">
        <div><span className="powerbi-eyebrow"><IconPresentation size={15} /> Espace décisionnel</span><h2>Bonjour {user?.name || user?.identifier}</h2><p>Accédez aux indicateurs de votre direction depuis un espace unique et sécurisé.</p></div>
        <div className="powerbi-access-badge">{visibleReports.length} tableau{visibleReports.length > 1 ? "x" : ""} accessible{visibleReports.length > 1 ? "s" : ""}</div>
      </div>
      {visibleReports.length > 1 && <div className="powerbi-tabs" role="tablist" aria-label="Choisir une direction">{visibleReports.map((report) => <button key={report.id} role="tab" aria-selected={selected.id === report.id} className={selected.id === report.id ? "active" : ""} onClick={() => setSelectedId(report.id)}><span>{report.shortLabel}</span></button>)}</div>}
      <article className="powerbi-frame-card">
        <div className="powerbi-frame-header"><div><h3>{selected.label}</h3><p>{selected.description}</p></div><a className="btn btn-secondary btn-sm" href={selected.url} target="_blank" rel="noopener noreferrer">Ouvrir dans Power BI <IconExternalLink size={15} /></a></div>
        <div className="powerbi-frame-wrap"><iframe key={selected.id} title={`Dashboard Power BI — ${selected.label}`} src={selected.url} allowFullScreen /></div>
        <p className="powerbi-frame-note">Une connexion Microsoft autorisée sur cet espace Power BI peut être demandée.</p>
      </article>
    </section>
  );
}
