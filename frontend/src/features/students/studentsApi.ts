import { apiRequest } from "@/services/api";
import type {
  ClassListResponse,
  NextRegistrationResponse,
  Student,
  StudentCreatePayload,
  StudentListResponse,
} from "@/types/students";

export function listStudents(token: string) {
  return apiRequest<StudentListResponse>("/students", { token });
}

export function listStudentClasses(token: string) {
  return apiRequest<ClassListResponse>("/students/classes", { token });
}

export function getNextRegistration(token: string) {
  return apiRequest<NextRegistrationResponse>("/students/next-registration", { token });
}

export function createStudent(token: string, payload: StudentCreatePayload) {
  return apiRequest<Student>("/students", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
  });
}

export function deactivateStudent(token: string, studentId: string) {
  return apiRequest<void>(`/students/${studentId}`, {
    method: "DELETE",
    token,
  });
}
