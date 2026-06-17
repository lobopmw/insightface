import { apiRequest } from "@/services/api";
import type { DashboardOverview } from "@/types/dashboard";

export function getDashboardOverview(token: string) {
  return apiRequest<DashboardOverview>("/dashboard/overview", { token });
}
