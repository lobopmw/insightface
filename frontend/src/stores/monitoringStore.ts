import { create } from "zustand";

import { closeMonitoringSession, createMonitoringSession } from "@/features/monitoring/monitoringApi";
import type { CreateMonitoringSessionPayload, MonitoringSession } from "@/types/monitoring";

type MonitoringState = {
  activeSession: MonitoringSession | null;
  isLoading: boolean;
  error: string | null;
  startSession: (token: string, payload: CreateMonitoringSessionPayload) => Promise<void>;
  stopSession: (token: string) => Promise<void>;
};

export const useMonitoringStore = create<MonitoringState>((set, get) => ({
  activeSession: null,
  isLoading: false,
  error: null,

  startSession: async (token, payload) => {
    set({ isLoading: true, error: null });
    try {
      const session = await createMonitoringSession(token, payload);
      set({ activeSession: session, isLoading: false });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : "Falha ao iniciar sessão",
      });
    }
  },

  stopSession: async (token) => {
    const activeSession = get().activeSession;
    if (!activeSession) {
      return;
    }
    set({ isLoading: true, error: null });
    try {
      const session = await closeMonitoringSession(token, activeSession.id);
      set({ activeSession: session, isLoading: false });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : "Falha ao encerrar sessão",
      });
    }
  },
}));
