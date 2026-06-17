import { useQuery } from "@tanstack/react-query";

import { getDashboardOverview } from "@/features/dashboard/dashboardApi";
import { useAuthStore } from "@/stores/authStore";

export function useDashboardOverview() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["dashboard", "overview"],
    queryFn: () => getDashboardOverview(token as string),
    enabled: Boolean(token),
  });
}
