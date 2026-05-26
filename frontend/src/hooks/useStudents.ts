import { useQuery } from "@tanstack/react-query";

import { getNextRegistration, listStudentClasses, listStudents } from "@/features/students/studentsApi";
import { useAuthStore } from "@/stores/authStore";

export function useStudents() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["students"],
    queryFn: () => listStudents(token as string),
    enabled: Boolean(token),
  });
}

export function useStudentClasses() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["students", "classes"],
    queryFn: () => listStudentClasses(token as string),
    enabled: Boolean(token),
  });
}

export function useNextRegistration() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["students", "next-registration"],
    queryFn: () => getNextRegistration(token as string),
    enabled: Boolean(token),
  });
}
