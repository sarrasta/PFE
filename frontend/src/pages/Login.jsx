import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../api/client";
import { IconLock } from "../components/Icons";

export function Login() {
  const { loginWithPassword, loginWithLink } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState("password"); // password | link
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [linkToken, setLinkToken] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const redirectTo = location.state?.from?.pathname || "/dashboard";

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "password") {
        await loginWithPassword(identifier, password);
      } else {
        await loginWithLink(linkToken);
      }
      navigate(redirectTo, { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setError("Aucun compte n'est encore configuré. Contactez votre administrateur.");
      } else {
        setError(err.message || "Échec de la connexion.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="login-panel">
        <div className="login-brand">
          <div className="sidebar-brand-mark" style={{ width: 44, height: 44, fontSize: 16 }}>TT</div>
          <div>
            <strong>Tunisie Telecom</strong>
            <p>Plateforme Analytique de Rétention Client</p>
          </div>
        </div>

        <h2>Connexion</h2>
        <p className="login-sub">Accédez au tableau de bord avec vos identifiants internes.</p>

        <div className="login-mode-toggle">
          <button
            type="button"
            className={mode === "password" ? "active" : ""}
            onClick={() => setMode("password")}
          >
            Identifiant + mot de passe
          </button>
          <button
            type="button"
            className={mode === "link" ? "active" : ""}
            onClick={() => setMode("link")}
          >
            Lien d'accès personnel
          </button>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {mode === "password" ? (
            <>
              <div className="field">
                <label htmlFor="identifier">Identifiant</label>
                <input
                  id="identifier"
                  className="input"
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="password">Mot de passe</label>
                <input
                  id="password"
                  type="password"
                  className="input"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </>
          ) : (
            <div className="field">
              <label htmlFor="linkToken">Jeton d'accès</label>
              <input
                id="linkToken"
                className="input"
                value={linkToken}
                onChange={(e) => setLinkToken(e.target.value)}
                placeholder="Collez le jeton reçu de votre administrateur"
                required
              />
            </div>
          )}

          {error && <div className="login-error">{error}</div>}

          <button type="submit" className="btn btn-primary" disabled={submitting} style={{ justifyContent: "center" }}>
            {submitting ? "Connexion..." : "Se connecter"}
          </button>
        </form>

        <div className="login-footnote">
          <IconLock size={13} /> Accès réservé au personnel autorisé de Tunisie Telecom.
        </div>
      </div>
      <div className="login-side" aria-hidden="true" />
    </div>
  );
}
