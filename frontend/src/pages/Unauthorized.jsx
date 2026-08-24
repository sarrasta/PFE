import { Link } from "react-router-dom";
import { IconLock } from "../components/Icons";

export function Unauthorized() {
  return (
    <div className="state-block" style={{ minHeight: "70vh" }}>
      <IconLock size={36} />
      <h4>Accès non autorisé</h4>
      <p>Votre rôle ne dispose pas des permissions nécessaires pour consulter cette page.</p>
      <Link to="/dashboard" className="btn btn-primary">Retour au tableau de bord</Link>
    </div>
  );
}
