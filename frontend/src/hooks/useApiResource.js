import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

/**
 * Fetches data from an async API call and exposes { data, loading, error,
 * warmingUp, reload }. `warmingUp` is set apart from a generic error when
 * the backend reports the ML pipeline is still training (HTTP 503,
 * error: "ml_pipeline_warming_up") — pages show a calm "training in
 * progress" state instead of a scary error in that case, and this hook
 * automatically retries every 8s while warming up.
 */
export function useApiResource(fetcher, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [warmingUp, setWarmingUp] = useState(false);
  const retryTimer = useRef(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetcher();
      setData(res);
      setWarmingUp(false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503 && err.payload?.error === "ml_pipeline_warming_up") {
        setWarmingUp(true);
        setError(null);
        clearTimeout(retryTimer.current);
        retryTimer.current = setTimeout(load, 8000);
      } else {
        setError(err.message || "Erreur inattendue.");
        setWarmingUp(false);
      }
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    load();
    return () => clearTimeout(retryTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  return { data, loading, error, warmingUp, reload: load };
}
