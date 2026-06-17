import { apiRequest } from "@/services/api";
import type { AdminCatalogResponse, AdminClass, AdminStudentListResponse, AdminSubject, AdminAssignment } from "@/types/adminCatalog";
import type { Student, StudentCreatePayload } from "@/types/student";

export function getAdminCatalog(token: string) {
  return apiRequest<AdminCatalogResponse>("/admin/catalog", { token });
}

export function createAdminSubject(token: string, nome: string) {
  return apiRequest<AdminSubject>("/admin/catalog/subjects", {
    body: JSON.stringify({ nome }),
    method: "POST",
    token,
  });
}

export function updateAdminSubject(token: string, subjectId: number, nome: string) {
  return apiRequest<AdminSubject>(`/admin/catalog/subjects/${subjectId}`, {
    body: JSON.stringify({ nome }),
    method: "PATCH",
    token,
  });
}

export function deleteAdminSubject(token: string, subjectId: number) {
  return apiRequest<void>(`/admin/catalog/subjects/${subjectId}`, { method: "DELETE", token });
}

export function createAdminClass(token: string, payload: { nome: string; identificador?: string | null }) {
  return apiRequest<AdminClass>("/admin/catalog/classes", {
    body: JSON.stringify(payload),
    method: "POST",
    token,
  });
}

export function updateAdminClass(token: string, classId: number, payload: { nome?: string; identificador?: string | null }) {
  return apiRequest<AdminClass>(`/admin/catalog/classes/${classId}`, {
    body: JSON.stringify(payload),
    method: "PATCH",
    token,
  });
}

export function deleteAdminClass(token: string, classId: number) {
  return apiRequest<void>(`/admin/catalog/classes/${classId}`, { method: "DELETE", token });
}

export function createAdminAssignment(token: string, payload: { teacher_id: number; subject_id: number; class_id: number }) {
  return apiRequest<AdminAssignment>("/admin/catalog/assignments", {
    body: JSON.stringify(payload),
    method: "POST",
    token,
  });
}

export function deleteAdminAssignment(token: string, assignmentId: number) {
  return apiRequest<void>(`/admin/catalog/assignments/${assignmentId}`, { method: "DELETE", token });
}

export function listAdminStudents(token: string) {
  return apiRequest<AdminStudentListResponse>("/admin/catalog/students", { token });
}

export function updateAdminStudent(token: string, studentId: string, payload: Partial<StudentCreatePayload> & { ativo?: boolean }) {
  return apiRequest<Student>(`/admin/catalog/students/${studentId}`, {
    body: JSON.stringify(payload),
    method: "PATCH",
    token,
  });
}
