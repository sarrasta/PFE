export function Card({ title, subtitle, actions, children, noPad = false, className = "" }) {
  return (
    <div className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-header">
          <div>
            {title && <h3>{title}</h3>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions && <div className="page-actions">{actions}</div>}
        </div>
      )}
      <div className={`card-body ${noPad ? "no-pad" : ""}`}>{children}</div>
    </div>
  );
}
