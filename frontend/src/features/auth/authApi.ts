import { apiRequest } from "@/services/api";
import type { CurrentUser, LoginCredentials, TokenResponse } from "@/types/auth";

export function loginRequest(credentials: LoginCredentials) {
  return apiRequest<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(credentials),
  });
}

export function getCurrentUser(token: string) {
  return apiRequest<CurrentUser>("/auth/me", { token });
}
