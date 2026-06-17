import { useQuery } from "@tanstack/react-query";

import { getReportSummary, listBehaviorEpisodes } from "@/features/reports/reportsApi";
import { useAuthStore } from "@/stores/authStore";

export function useReportSummary() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["reports", "summary"],
    queryFn: () => getReportSummary(token as string),
    enabled: Boolean(token),
  });
}

export function useBehaviorEpisodes() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["reports", "episodes"],
    queryFn: () => listBehaviorEpisodes(token as string),
    enabled: Boolean(token),
  });
}
