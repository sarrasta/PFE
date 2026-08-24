import { AdminApi } from "../api/endpoints";
import { useApiResource } from "../hooks/useApiResource";
import { AsyncSection, EmptyState } from "../components/States";
import { Card } from "../components/Card";
import { Badge } from "../components/Badge";
import { IconCheckCircle, IconUsers, IconXCircle } from "../components/Icons";

export function Settings() {
  const { data, loading, error, warmingUp, reload } = useApiResource(AdminApi.settings, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Paramètres</h2>
          <p>Rôles, permissions, paramètres du moteur ML et répertoire des comptes</p>
        </div>
      </div>

      <AsyncSection loading={loading} warmingUp={warmingUp} error={error} onRetry={reload}>
        {data && (
          <>
            <div className="page-grid grid-2" style={{ marginBottom: "var(--space-6)" }}>
              <Card title="Paramètres du moteur ML" subtitle="Objectif 9 — formule de gain">
                <SettingRow label="Horizon LTV" value={`${data.ml_parameters.retention_ltv_horizon_months} mois`} />
                <SettingRow label="Coût de l'offre par défaut" value={`${data.ml_parameters.retention_offer_cost_tnd} TND`} />
                <SettingRow
                  label="Réentraînement automatique"
                  value={data.ml_parameters.auto_refresh_interval_minutes > 0
                    ? `Toutes les ${data.ml_parameters.auto_refresh_interval_minutes} min`
                    : "Désactivé (manuel via Monitoring)"}
                />
              </Card>

              <Card title="Comptes configurés" subtitle={<span style={{ display: "flex", alignItems: "center", gap: 5 }}><IconUsers size={13} /> {data.users.length} / 5 comptes actifs</span>}>
                {data.users.length === 0 ? (
                  <EmptyState
                    title="Aucun compte configuré"
                    message="Renseignez ADMIN_USER/ADMIN_PASSWORD et USER_1..USER_4 dans l'environnement du backend — voir .env.example."
                  />
                ) : (
                  data.users.map((u) => (
                    <div key={u.identifier} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid var(--border-hairline)" }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{u.name}</div>
                        <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>{u.identifier}</div>
                      </div>
                      <Badge tone={u.role === "admin" ? "brand" : "neutral"}>{u.role}</Badge>
                    </div>
                  ))
                )}
                <p style={{ fontSize: 11.5, color: "var(--text-muted)", marginTop: "var(--space-3)" }}>
                  Pour ajouter ou modifier des comptes : mettez à jour les variables ADMIN_USER / USER_1..USER_4
                  (et leurs mots de passe ou jetons d'accès) dans le fichier .env, puis redémarrez le conteneur backend.
                </p>
              </Card>
            </div>

            <Card title="Matrice des permissions par rôle" noPad>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Section</th>
                      <th>Admin</th>
                      <th>Utilisateur standard</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.all_scopes.map((scope) => (
                      <tr key={scope}>
                        <td style={{ textTransform: "capitalize" }}>{scope}</td>
                        <td><IconCheckCircle size={16} style={{ color: "var(--status-good)" }} /></td>
                        <td>
                          {data.role_permissions.user?.includes(scope)
                            ? <IconCheckCircle size={16} style={{ color: "var(--status-good)" }} />
                            : <IconXCircle size={16} style={{ color: "var(--text-muted)" }} />}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </AsyncSection>
    </div>
  );
}

function SettingRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid var(--border-hairline)", fontSize: 13 }}>
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
