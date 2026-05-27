import { useQuery } from "@tanstack/react-query";

import { getAdminCatalog, listAdminStudents } from "@/features/admin/catalogApi";
import { useAuthStore } from "@/stores/authStore";

export function useAdminCatalog() {
  const token = useAuthStore((state) => state.token);
  const user = useAuthStore((state) => state.user);

  return useQuery({
    queryKey: ["admin-catalog"],
    queryFn: () => getAdminCatalog(token as string),
    enabled: Boolean(token && user?.role === "admin"),
  });
}

export function useAdminStudents() {
  const token = useAuthStore((state) => state.token);
  const user = useAuthStore((state) => state.user);

  return useQuery({
    queryKey: ["admin-students"],
    queryFn: () => listAdminStudents(token as string),
    enabled: Boolean(token && user?.role === "admin"),
  });
}
