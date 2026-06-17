import { useQuery } from "@tanstack/react-query";

import { apiRequest } from "@/services/api";
import type { HealthResponse } from "@/types/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiRequest<HealthResponse>("/health"),
    refetchInterval: 30000,
  });
}
