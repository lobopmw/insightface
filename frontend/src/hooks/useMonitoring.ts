import { useCallback, useEffect, useMemo, useState } from "react";

import { getMonitoringStatus, startMonitoring, stopMonitoring } from "@/features/monitoring/monitoringApi";
import { createMonitoringSocket } from "@/services/websocket";
import { useAuthStore } from "@/stores/authStore";
import type { MonitoringRealtimeEvent, MonitoringStartPayload, MonitoringStatus } from "@/types/monitoring";

const initialStatus: MonitoringStatus = {
  status: "idle",
  session_id: null,
  camera_status: "idle",
  websocket_clients: 0,
  student_name: "Aguardando identificação",
  behavior: "Aguardando",
  confidence: 0,
  recognized_students_now: 0,
};

function isRecognizedStudentName(name?: string) {
  const normalized = (name ?? "").trim().toLowerCase();
  return Boolean(
    normalized &&
      normalized !== "desconhecido" &&
      normalized !== "aguardando identificação" &&
      normalized !== "aguardando identificacao",
  );
}

export function useMonitoring() {
  const token = useAuthStore((state) => state.token);
  const [status, setStatus] = useState<MonitoringStatus>(initialStatus);
  const [events, setEvents] = useState<MonitoringRealtimeEvent[]>([]);
  const [websocketStatus, setWebsocketStatus] = useState<"connecting" | "connected" | "disconnected" | "error">(
    "disconnected",
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    if (!token) {
      return;
    }
    try {
      const nextStatus = await getMonitoringStatus(token);
      setStatus(nextStatus);
    } catch (statusError) {
      setError(statusError instanceof Error ? statusError.message : "Falha ao consultar status");
    }
  }, [token]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    if (!token) {
      return;
    }

    setWebsocketStatus("connecting");
    const socket = createMonitoringSocket(token);

    socket.addEventListener("open", () => setWebsocketStatus("connected"));
    socket.addEventListener("error", () => setWebsocketStatus("error"));
    socket.addEventListener("close", () => setWebsocketStatus("disconnected"));
    socket.addEventListener("message", (message) => {
      try {
        const parsed = JSON.parse(message.data as string) as MonitoringRealtimeEvent;
        const payload = parsed.payload ?? {};
        const normalizedEvent: MonitoringRealtimeEvent = {
          ...parsed,
          session_id: parsed.session_id ?? (payload.session_id as number | string | null | undefined),
          student_name: parsed.student_name ?? (payload.student_name as string | undefined),
          behavior: parsed.behavior ?? (payload.behavior as string | undefined),
          confidence: parsed.confidence ?? (payload.confidence as number | undefined),
          faces_detected: parsed.faces_detected ?? (payload.faces_detected as number | undefined),
          people_detected: parsed.people_detected ?? (payload.people_detected as number | undefined),
          recognized_students_now:
            parsed.recognized_students_now ?? (payload.recognized_students_now as number | undefined),
          camera_status: parsed.camera_status ?? (payload.camera_status as string | undefined),
          websocket_status: parsed.websocket_status ?? (payload.websocket_status as string | undefined),
          timestamp: parsed.timestamp ?? (payload.timestamp as string | undefined),
        };

        setEvents((current) => [normalizedEvent, ...current].slice(0, 20));
        if (normalizedEvent.type === "behavior_event") {
          setStatus((current) => ({
            ...current,
            session_id: Number(normalizedEvent.session_id ?? current.session_id),
            student_name: normalizedEvent.student_name ?? current.student_name,
            behavior: normalizedEvent.behavior ?? current.behavior,
            confidence: normalizedEvent.confidence ?? current.confidence,
            recognized_students_now:
              normalizedEvent.recognized_students_now ??
              (isRecognizedStudentName(normalizedEvent.student_name) ? 1 : 0),
            camera_status: normalizedEvent.camera_status ?? current.camera_status,
            timestamp: normalizedEvent.timestamp ?? current.timestamp,
          }));
        }
      } catch {
        setEvents((current) => [{ type: "message", timestamp: new Date().toISOString() }, ...current].slice(0, 20));
      }
    });

    return () => {
      socket.close();
    };
  }, [token]);

  const start = useCallback(
    async (payload: MonitoringStartPayload) => {
      if (!token) {
        return;
      }
      setIsLoading(true);
      setError(null);
      try {
        const nextStatus = await startMonitoring(token, payload);
        setStatus(nextStatus);
      } catch (startError) {
        setError(startError instanceof Error ? startError.message : "Falha ao iniciar monitoramento");
      } finally {
        setIsLoading(false);
      }
    },
    [token],
  );

  const stop = useCallback(async () => {
    if (!token) {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const nextStatus = await stopMonitoring(token);
      setStatus(nextStatus);
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : "Falha ao parar monitoramento");
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  const isActive = useMemo(() => status.status === "running" && Boolean(status.session_id), [status]);

  return {
    error,
    events,
    isActive,
    isLoading,
    refreshStatus,
    start,
    status,
    stop,
    websocketStatus,
  };
}
