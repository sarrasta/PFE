import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { LoadingState } from "./States";

export function RequireAuth({ children }) {
  const { status } = useAuth();
  const location = useLocation();

  if (status === "checking") return <LoadingState label="Vérification de la session..." />;
  if (status === "anonymous") return <Navigate to="/login" state={{ from: location }} replace />;
  return children;
}

export function RequireScope({ scope, children }) {
  const { hasScope } = useAuth();
  if (!hasScope(scope)) return <Navigate to="/unauthorized" replace />;
  return children;
}

export function RequireAnyScope({ scopes, children }) {
  const { hasScope } = useAuth();
  if (!scopes.some(hasScope)) return <Navigate to="/unauthorized" replace />;
  return children;
}
