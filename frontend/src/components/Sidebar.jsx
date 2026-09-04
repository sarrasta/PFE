import { NavLink } from "react-router-dom";
import { NAV_ADMIN_ITEMS, NAV_DIRECTION_ITEM, NAV_ITEMS } from "../navConfig";
import { useAuth } from "../context/AuthContext";
import { IconChevronLeft, IconChevronRight } from "./Icons";

function NavItem({ item, collapsed, onNavigate }) {
  return <NavLink to={item.to} onClick={onNavigate} className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`} title={collapsed ? item.label : undefined}>
    <item.icon size={18} />{!collapsed && <span className="sidebar-link-label">{item.label}</span>}
  </NavLink>;
}

export function Sidebar({ collapsed, onToggle, onNavigate }) {
  const { hasScope } = useAuth();
  const visibleItems = NAV_ITEMS.filter((item) => hasScope(item.scope));
  const visibleAdminItems = NAV_ADMIN_ITEMS.filter((item) => hasScope(item.scope));
  const showDirections = NAV_DIRECTION_ITEM.scopes.some(hasScope);

  return <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
    <div className="sidebar-brand"><div className="sidebar-brand-mark"><span>TT</span></div>{!collapsed && <div className="sidebar-brand-text"><strong>Tunisie Telecom</strong><span>Data &amp; Intelligence</span></div>}</div>
    <nav className="sidebar-nav">
      {visibleItems.length > 0 && <>{!collapsed && <div className="sidebar-section-label">Analytique</div>}{visibleItems.map((item) => <NavItem key={item.to} item={item} collapsed={collapsed} onNavigate={onNavigate} />)}</>}
      {showDirections && <>{!collapsed && <div className="sidebar-section-label">Pilotage métier</div>}<NavItem item={NAV_DIRECTION_ITEM} collapsed={collapsed} onNavigate={onNavigate} /></>}
      {visibleAdminItems.length > 0 && <>{!collapsed && <div className="sidebar-section-label">Administration</div>}{visibleAdminItems.map((item) => <NavItem key={item.to} item={item} collapsed={collapsed} onNavigate={onNavigate} />)}</>}
    </nav>
    <div className="sidebar-footer"><button className="sidebar-collapse-btn" onClick={onToggle}>{collapsed ? <IconChevronRight size={16} /> : <><IconChevronLeft size={16} /> Réduire</>}</button></div>
  </aside>;
}
