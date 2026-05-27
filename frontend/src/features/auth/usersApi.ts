import { apiRequest } from "@/services/api";
import type { AdminUser, AdminUserCreatePayload, AdminUserListResponse, AdminUserUpdatePayload } from "@/types/users";

export function listAdminUsers(token: string) {
  return apiRequest<AdminUserListResponse>("/users", { token });
}

export function createAdminUser(token: string, payload: AdminUserCreatePayload) {
  return apiRequest<AdminUser>("/users", {
    body: JSON.stringify(payload),
    method: "POST",
    token,
  });
}

export function updateAdminUser(token: string, userId: number, payload: AdminUserUpdatePayload) {
  return apiRequest<AdminUser>(`/users/${userId}`, {
    body: JSON.stringify(payload),
    method: "PATCH",
    token,
  });
}
