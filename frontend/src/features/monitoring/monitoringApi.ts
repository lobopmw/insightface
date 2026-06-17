import { apiRequest } from "@/services/api";
import type {
  CreateMonitoringSessionPayload,
  MonitoringStartPayload,
  MonitoringStatus,
  MonitoringOptions,
  MonitoringSession,
} from "@/types/monitoring";

export function getMonitoringOptions(token: string) {
  return apiRequest<MonitoringOptions>("/monitoring/options", { token });
}

export function createMonitoringSession(token: string, payload: CreateMonitoringSessionPayload) {
  return apiRequest<MonitoringSession>("/monitoring/sessions", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
  });
}

export function closeMonitoringSession(token: string, sessionId: number) {
  return apiRequest<MonitoringSession>(`/monitoring/sessions/${sessionId}/close`, {
    method: "POST",
    token,
    body: JSON.stringify({ status: "encerrada" }),
  });
}

export function getMonitoringStatus(token: string) {
  return apiRequest<MonitoringStatus>("/monitoring/status", { token });
}

export function startMonitoring(token: string, payload: MonitoringStartPayload) {
  return apiRequest<MonitoringStatus>("/monitoring/start", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
  });
}

export function stopMonitoring(token: string) {
  return apiRequest<MonitoringStatus>("/monitoring/stop", {
    method: "POST",
    token,
    body: JSON.stringify({ status: "encerrada" }),
  });
}
