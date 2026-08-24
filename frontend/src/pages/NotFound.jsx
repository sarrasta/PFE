import { Link } from "react-router-dom";
import { IconAlertTriangle } from "../components/Icons";

export function NotFound() {
  return (
    <div className="state-block" style={{ minHeight: "70vh" }}>
      <IconAlertTriangle size={36} />
      <h4>Page introuvable</h4>
      <p>La page que vous cherchez n'existe pas ou a été déplacée.</p>
      <Link to="/dashboard" className="btn btn-primary">Retour au tableau de bord</Link>
    </div>
  );
}
