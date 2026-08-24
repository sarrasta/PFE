import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { NAV_ADMIN_ITEMS, NAV_ITEMS } from "../navConfig";

const ALL_ITEMS = [...NAV_ITEMS, ...NAV_ADMIN_ITEMS];

export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const current = ALL_ITEMS.find((item) => location.pathname.startsWith(item.to));

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <div className="app-main">
        <Topbar title={current?.label || "Tableau de bord"} onMenuClick={() => setCollapsed((c) => !c)} />
        <main className="app-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
