import { useQuery } from "@tanstack/react-query";

import { getNextRegistration, getStudentFaceStatus, listStudentClasses, listStudents } from "@/services/studentsApi";
import { useAuthStore } from "@/stores/authStore";

export function useStudents() {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["students"],
    queryFn: () => listStudents(token as string),
    enabled: Boolean(token),
  });
}

export function useStudentFaceStatus(studentId?: string | null) {
  const token = useAuthStore((state) => state.token);

  return useQuery({
    queryKey: ["students", studentId, "face-status"],
    queryFn: () => getStudentFaceStatus(token as string, studentId as string),
    enabled: Boolean(token && studentId),
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
