import { apiRequest } from "@/services/api";
import type { BehaviorEpisodeListResponse, ReportSummaryResponse } from "@/types/reports";

export function getReportSummary(token: string) {
  return apiRequest<ReportSummaryResponse>("/reports/summary", { token });
}

export function listBehaviorEpisodes(token: string) {
  return apiRequest<BehaviorEpisodeListResponse>("/reports/episodes?limit=50", { token });
}
