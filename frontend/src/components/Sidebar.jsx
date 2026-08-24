import { NavLink } from "react-router-dom";
import { NAV_ADMIN_ITEMS, NAV_ITEMS } from "../navConfig";
import { useAuth } from "../context/AuthContext";
import { IconChevronLeft, IconChevronRight } from "./Icons";

export function Sidebar({ collapsed, onToggle }) {
  const { hasScope } = useAuth();
  const visibleItems = NAV_ITEMS.filter((item) => hasScope(item.scope));
  const visibleAdminItems = NAV_ADMIN_ITEMS.filter((item) => hasScope(item.scope));

  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">TT</div>
        {!collapsed && (
          <div className="sidebar-brand-text">
            <strong>Tunisie Telecom</strong>
            <span>Analytique Rétention</span>
          </div>
        )}
      </div>

      <nav className="sidebar-nav">
        {!collapsed && <div className="sidebar-section-label">Analytique</div>}
        {visibleItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}
            title={collapsed ? item.label : undefined}
          >
            <item.icon size={17} />
            {!collapsed && <span className="sidebar-link-label">{item.label}</span>}
          </NavLink>
        ))}

        {visibleAdminItems.length > 0 && (
          <>
            {!collapsed && <div className="sidebar-section-label">Administration</div>}
            {visibleAdminItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}
                title={collapsed ? item.label : undefined}
              >
                <item.icon size={17} />
                {!collapsed && <span className="sidebar-link-label">{item.label}</span>}
              </NavLink>
            ))}
          </>
        )}
      </nav>

      <div className="sidebar-footer">
        <button className="sidebar-collapse-btn" onClick={onToggle}>
          {collapsed ? <IconChevronRight size={16} /> : <><IconChevronLeft size={16} /> Réduire</>}
        </button>
      </div>
    </aside>
  );
}
