import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { usePipelineStatus } from "../hooks/usePipelineStatus";
import { IconLogOut, IconMenu } from "./Icons";

const STATUS_LABEL = {
  ready: "Modèles à jour",
  loading: "Entraînement en cours…",
  error: "Erreur pipeline ML",
  not_initialized: "Initialisation…",
};

export function Topbar({ title, subtitle, onMenuClick }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const health = usePipelineStatus();
  const pipelineStatus = health?.ml_pipeline?.status || "not_initialized";
  const dotClass = pipelineStatus === "ready" ? "dot-ready" : pipelineStatus === "error" ? "dot-error" : "dot-loading";

  async function handleLogout() {
    await logout();
    navigate("/login");
  }

  const initials = (user?.name || user?.identifier || "?")
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <header className="topbar">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button className="icon-btn" onClick={onMenuClick} aria-label="Basculer le menu">
          <IconMenu size={18} />
        </button>
        <div className="topbar-title">
          <h1>{title}</h1>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>

      <div className="topbar-right">
        <span className="pipeline-pill" title={health?.ml_pipeline?.last_updated ? `Dernière mise à jour : ${new Date(health.ml_pipeline.last_updated).toLocaleString("fr-FR")}` : undefined}>
          <span className={`dot ${dotClass}`} />
          {STATUS_LABEL[pipelineStatus] || pipelineStatus}
        </span>

        <div className="user-menu">
          <div className="user-avatar">{initials}</div>
          <div className="user-meta">
            <strong>{user?.name || user?.identifier}</strong>
            <span>{user?.role}</span>
          </div>
          <button className="icon-btn" onClick={handleLogout} aria-label="Se déconnecter" title="Se déconnecter">
            <IconLogOut size={17} />
          </button>
        </div>
      </div>
    </header>
  );
}
