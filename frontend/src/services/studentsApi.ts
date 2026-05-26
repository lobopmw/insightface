import { apiRequest } from "@/services/api";
import type {
  ClassListResponse,
  EmbeddingGenerationResponse,
  FaceImageUploadResponse,
  NextRegistrationResponse,
  PoseKey,
  Student,
  StudentCreatePayload,
  StudentFaceStatus,
  StudentListResponse,
} from "@/types/student";

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

export function updateStudent(token: string, studentId: string, payload: Partial<StudentCreatePayload>) {
  return apiRequest<Student>(`/students/${studentId}`, {
    method: "PUT",
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

export function getStudentFaceStatus(token: string, studentId: string) {
  return apiRequest<StudentFaceStatus>(`/students/${studentId}/face-status`, { token });
}

export function uploadStudentFaceImages(token: string, studentId: string, pose: PoseKey, files: Blob[]) {
  const formData = new FormData();
  formData.set("pose", pose);
  files.forEach((file, index) => {
    formData.append("files", file, `${pose}_${index + 1}.jpg`);
  });
  return apiRequest<FaceImageUploadResponse>(`/students/${studentId}/face-images`, {
    method: "POST",
    token,
    body: formData,
  });
}

export function uploadStudentFaceImage(token: string, studentId: string, pose: PoseKey, image: Blob, index: number) {
  const formData = new FormData();
  formData.set("pose", pose);
  formData.set("index", String(index));
  formData.set("image", image, `${pose}_${index}.jpg`);
  return apiRequest<FaceImageUploadResponse>(`/students/${studentId}/face-images`, {
    method: "POST",
    token,
    body: formData,
  });
}

export function generateStudentEmbeddings(token: string, studentId: string) {
  return apiRequest<EmbeddingGenerationResponse>(`/students/${studentId}/generate-embeddings`, {
    method: "POST",
    token,
  });
}
