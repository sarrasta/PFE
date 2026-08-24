import {
  IconActivity, IconBarChart, IconDashboard, IconLayers,
  IconSettings, IconTarget, IconTrendDown, IconUsers, IconWallet,
} from "./components/Icons";

export const NAV_ITEMS = [
  { to: "/dashboard", label: "Tableau de bord", icon: IconDashboard, scope: "dashboard" },
  { to: "/clients", label: "Clients", icon: IconUsers, scope: "clients" },
  { to: "/churn-risque", label: "Churn & Risque", icon: IconTrendDown, scope: "churn" },
  { to: "/segmentation", label: "Segmentation", icon: IconLayers, scope: "segmentation" },
  { to: "/retention", label: "Rétention", icon: IconTarget, scope: "retention" },
  { to: "/revenus", label: "Revenus & ARPU", icon: IconWallet, scope: "revenue" },
  { to: "/performance", label: "Performance des modèles", icon: IconBarChart, scope: "models" },
];

export const NAV_ADMIN_ITEMS = [
  { to: "/monitoring", label: "Monitoring", icon: IconActivity, scope: "monitoring" },
  { to: "/parametres", label: "Paramètres", icon: IconSettings, scope: "settings" },
];
