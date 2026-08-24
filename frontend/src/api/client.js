const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";
const TOKEN_KEY = "tt_retention_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

let onUnauthorized = null;
export function registerUnauthorizedHandler(fn) {
  onUnauthorized = fn;
}

async function request(path, { method = "GET", body, params, skipAuth = false } = {}) {
  const url = new URL(API_BASE + path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
  }

  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token && !skipAuth) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(url.pathname + url.search, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (networkErr) {
    throw new ApiError("Impossible de contacter le serveur. Vérifiez votre connexion.", 0, null);
  }

  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (response.status === 401 && !skipAuth) {
    onUnauthorized?.();
  }

  if (!response.ok) {
    const message = payload?.message || `Erreur ${response.status}`;
    throw new ApiError(message, response.status, payload);
  }

  return payload;
}

export const api = {
  get: (path, params) => request(path, { method: "GET", params }),
  post: (path, body, opts = {}) => request(path, { method: "POST", body, ...opts }),
};

export { ApiError };
