import { useQuery } from "@tanstack/react-query";

import { getMonitoringOptions } from "@/features/monitoring/monitoringApi";
import { useAuthStore } from "@/stores/authStore";

export function useMonitoringOptions() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["monitoring", "options"],
    queryFn: () => getMonitoringOptions(token as string),
    enabled: Boolean(token),
  });
}
