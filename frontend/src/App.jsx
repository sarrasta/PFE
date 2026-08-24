import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { RequireAuth, RequireScope } from "./components/ProtectedRoute";
import { AppLayout } from "./components/AppLayout";
import { Login } from "./pages/Login";
import { Unauthorized } from "./pages/Unauthorized";
import { NotFound } from "./pages/NotFound";
import { Dashboard } from "./pages/Dashboard";
import { Clients } from "./pages/Clients";
import { ClientDetail } from "./pages/ClientDetail";
import { ChurnRisk } from "./pages/ChurnRisk";
import { Segmentation } from "./pages/Segmentation";
import { Retention } from "./pages/Retention";
import { Revenue } from "./pages/Revenue";
import { ModelPerformance } from "./pages/ModelPerformance";
import { Monitoring } from "./pages/Monitoring";
import { Settings } from "./pages/Settings";

function Guarded({ scope, children }) {
  return (
    <RequireAuth>
      <RequireScope scope={scope}>{children}</RequireScope>
    </RequireAuth>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/unauthorized" element={<RequireAuth><Unauthorized /></RequireAuth>} />

        <Route element={<RequireAuth><AppLayout /></RequireAuth>}>
          <Route path="/dashboard" element={<Guarded scope="dashboard"><Dashboard /></Guarded>} />
          <Route path="/clients" element={<Guarded scope="clients"><Clients /></Guarded>} />
          <Route path="/clients/:clientId" element={<Guarded scope="clients"><ClientDetail /></Guarded>} />
          <Route path="/churn-risque" element={<Guarded scope="churn"><ChurnRisk /></Guarded>} />
          <Route path="/segmentation" element={<Guarded scope="segmentation"><Segmentation /></Guarded>} />
          <Route path="/retention" element={<Guarded scope="retention"><Retention /></Guarded>} />
          <Route path="/revenus" element={<Guarded scope="revenue"><Revenue /></Guarded>} />
          <Route path="/performance" element={<Guarded scope="models"><ModelPerformance /></Guarded>} />
          <Route path="/monitoring" element={<Guarded scope="monitoring"><Monitoring /></Guarded>} />
          <Route path="/parametres" element={<Guarded scope="settings"><Settings /></Guarded>} />
        </Route>

        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </AuthProvider>
  );
}
