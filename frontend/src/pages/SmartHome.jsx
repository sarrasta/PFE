import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { NAV_ITEMS, POWER_BI_SCOPES } from "../navConfig";

export function SmartHome() {
  const { hasScope } = useAuth();
  const firstAnalyticsPage = NAV_ITEMS.find((item) => hasScope(item.scope));
  if (firstAnalyticsPage) return <Navigate to={firstAnalyticsPage.to} replace />;
  if (POWER_BI_SCOPES.some(hasScope)) return <Navigate to="/directions" replace />;
  return <Navigate to="/unauthorized" replace />;
}
