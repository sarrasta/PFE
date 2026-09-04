import { api } from "./client";

export const AuthApi = {
  login: (identifier, password) => api.post("/auth/login", { identifier, password }, { skipAuth: true }),
  linkLogin: (token) => api.post("/auth/link-login", { token }, { skipAuth: true }),
  me: () => api.get("/auth/me"),
  logout: () => api.post("/auth/logout"),
};

export const HealthApi = {
  check: () => api.get("/health"),
};

export const DashboardApi = {
  get: () => api.get("/dashboard"),
};

export const ClientsApi = {
  list: (params) => api.get("/clients", params),
  filters: () => api.get("/clients/filters"),
  get: (id) => api.get(`/clients/${id}`),
};

export const ChurnApi = {
  overview: () => api.get("/churn/overview"),
  topRisk: (limit) => api.get("/churn/top-risk", { limit }),
};

export const SegmentationApi = {
  overview: () => api.get("/segmentation/overview"),
};

export const RetentionApi = {
  targetingMatrix: () => api.get("/retention/targeting-matrix"),
  gainOverview: () => api.get("/retention/gain-overview"),
  simulateScenario: (ltvHorizonMonths, offerCostTnd) =>
    api.post("/retention/simulate-scenario", { ltv_horizon_months: ltvHorizonMonths, offer_cost_tnd: offerCostTnd }),
};

export const RevenueApi = {
  overview: () => api.get("/revenue/overview"),
};

export const ModelsApi = {
  list: () => api.get("/models"),
  get: (id) => api.get(`/models/${id}`),
  predictCustomer: (stats, monthlyHistory) => api.post("/predict/customer", { stats, monthly_history: monthlyHistory }),
};

export const AdminApi = {
  monitoring: () => api.get("/monitoring"),
  settings: () => api.get("/settings"),
  refresh: () => api.post("/admin/refresh"),
  users: () => api.get("/admin/users"),
};
