import { useQuery } from "@tanstack/react-query";

import { listAdminUsers } from "@/features/auth/usersApi";
import { useAuthStore } from "@/stores/authStore";

export function useAdminUsers() {
  const token = useAuthStore((state) => state.token);
  const user = useAuthStore((state) => state.user);

  return useQuery({
    queryKey: ["admin-users"],
    queryFn: () => listAdminUsers(token as string),
    enabled: Boolean(token && user?.role === "admin"),
  });
}
