import { apiRequest } from "@/services/api";
import type { AdminUserListResponse } from "@/types/users";

export function listAdminUsers(token: string) {
  return apiRequest<AdminUserListResponse>("/users", { token });
}
