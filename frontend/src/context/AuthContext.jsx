import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { AuthApi } from "../api/endpoints";
import { getToken, registerUnauthorizedHandler, setToken } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [scopes, setScopes] = useState([]);
  const [status, setStatus] = useState("checking"); // checking | authenticated | anonymous

  const clearSession = useCallback(() => {
    setToken(null);
    setUser(null);
    setScopes([]);
    setStatus("anonymous");
  }, []);

  useEffect(() => {
    registerUnauthorizedHandler(clearSession);
  }, [clearSession]);

  useEffect(() => {
    const token = getToken();
    if (!token) {
      setStatus("anonymous");
      return;
    }
    AuthApi.me()
      .then((res) => {
        setUser(res.user);
        setScopes(res.scopes || []);
        setStatus("authenticated");
      })
      .catch(() => clearSession());
  }, [clearSession]);

  const loginWithPassword = useCallback(async (identifier, password) => {
    const res = await AuthApi.login(identifier, password);
    setToken(res.token);
    setUser(res.user);
    const me = await AuthApi.me();
    setScopes(me.scopes || []);
    setStatus("authenticated");
    return res.user;
  }, []);

  const loginWithLink = useCallback(async (linkToken) => {
    const res = await AuthApi.linkLogin(linkToken);
    setToken(res.token);
    setUser(res.user);
    const me = await AuthApi.me();
    setScopes(me.scopes || []);
    setStatus("authenticated");
    return res.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await AuthApi.logout();
    } catch {
      // ignore — logging out locally regardless
    }
    clearSession();
  }, [clearSession]);

  const hasScope = useCallback(
    (scope) => scopes.includes("*") || scopes.includes(scope),
    [scopes]
  );

  const value = useMemo(
    () => ({ user, scopes, status, loginWithPassword, loginWithLink, logout, hasScope }),
    [user, scopes, status, loginWithPassword, loginWithLink, logout, hasScope]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
