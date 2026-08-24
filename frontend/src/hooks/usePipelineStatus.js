import { useEffect, useRef, useState } from "react";
import { HealthApi } from "../api/endpoints";

/** Polls /api/health on an interval so the topbar pill and gates around the
 * app reflect the ML pipeline's real state without the user refreshing. */
export function usePipelineStatus(intervalMs = 20000) {
  const [health, setHealth] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const res = await HealthApi.check();
        if (!cancelled) setHealth(res);
      } catch {
        if (!cancelled) setHealth((prev) => ({ ...(prev || {}), status: "degraded" }));
      }
    }

    poll();
    timer.current = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer.current);
    };
  }, [intervalMs]);

  return health;
}
