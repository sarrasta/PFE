import {
  IconActivity, IconBarChart, IconDashboard, IconLayers,
  IconSettings, IconTarget, IconTrendDown, IconUsers, IconWallet, IconPresentation,
} from "./components/Icons";

export const POWER_BI_SCOPES = ["powerbi_marketing", "powerbi_reseau", "powerbi_financiere", "powerbi_regionale"];

export const NAV_ITEMS = [
  { to: "/dashboard", label: "Tableau de bord", icon: IconDashboard, scope: "dashboard" },
  { to: "/clients", label: "Clients", icon: IconUsers, scope: "clients" },
  { to: "/churn-risque", label: "Churn & Risque", icon: IconTrendDown, scope: "churn" },
  { to: "/segmentation", label: "Segmentation", icon: IconLayers, scope: "segmentation" },
  { to: "/retention", label: "Rétention", icon: IconTarget, scope: "retention" },
  { to: "/revenus", label: "Revenus & ARPU", icon: IconWallet, scope: "revenue" },
  { to: "/performance", label: "Prédiction ML", icon: IconBarChart, scope: "models" },
];

export const NAV_DIRECTION_ITEM = { to: "/directions", label: "Dashboards Directions", icon: IconPresentation, scopes: POWER_BI_SCOPES };

export const NAV_ADMIN_ITEMS = [
  { to: "/monitoring", label: "Monitoring", icon: IconActivity, scope: "monitoring" },
  { to: "/parametres", label: "Paramètres", icon: IconSettings, scope: "settings" },
];
