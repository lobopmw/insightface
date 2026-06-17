import { create } from "zustand";

import { getCurrentUser, loginRequest } from "@/features/auth/authApi";
import type { CurrentUser, LoginCredentials } from "@/types/auth";

const TOKEN_STORAGE_KEY = "insightface_access_token";

type AuthState = {
  token: string | null;
  user: CurrentUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginCredentials) => Promise<void>;
  restoreSession: () => Promise<void>;
  logout: () => void;
};

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem(TOKEN_STORAGE_KEY),
  user: null,
  isAuthenticated: Boolean(localStorage.getItem(TOKEN_STORAGE_KEY)),
  isLoading: false,
  error: null,

  login: async (credentials) => {
    set({ isLoading: true, error: null });
    try {
      const tokenResponse = await loginRequest(credentials);
      localStorage.setItem(TOKEN_STORAGE_KEY, tokenResponse.access_token);
      const user = await getCurrentUser(tokenResponse.access_token);
      set({
        token: tokenResponse.access_token,
        user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      set({
        token: null,
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: error instanceof Error ? error.message : "Falha ao autenticar",
      });
    }
  },

  restoreSession: async () => {
    const token = get().token;
    if (!token || get().user) {
      return;
    }
    set({ isLoading: true, error: null });
    try {
      const user = await getCurrentUser(token);
      set({ user, isAuthenticated: true, isLoading: false });
    } catch {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      set({ token: null, user: null, isAuthenticated: false, isLoading: false });
    }
  },

  logout: () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    set({ token: null, user: null, isAuthenticated: false, error: null });
  },
}));
