export type Student = {
  id: string;
  name: string;
  matricula?: string | null;
  class_id?: number | null;
  class_name?: string | null;
  class_identifier?: string | null;
};

export type StudentListResponse = {
  items: Student[];
};

export type StudentClass = {
  id: number;
  nome: string;
  identificador?: string | null;
};

export type ClassListResponse = {
  items: StudentClass[];
};

export type NextRegistrationResponse = {
  matricula: string;
};

export type StudentCreatePayload = {
  name: string;
  matricula?: string | null;
  class_id?: number | null;
};

export type PoseKey = "frontal" | "lateral_esquerda" | "lateral_direita" | "cabeca_baixa";

export type PoseStatus = {
  count: number;
  complete: boolean;
};

export type StudentFaceStatus = Record<PoseKey, PoseStatus> & {
  student_id: string;
  embeddings_generated: boolean;
};

export type StudentFaceStatusListResponse = {
  items: StudentFaceStatus[];
};

export type FaceImageUploadResponse = {
  student_id: string;
  pose: PoseKey;
  saved: number;
  status: StudentFaceStatus;
};

export type EmbeddingGenerationResponse = {
  student_id: string;
  success: boolean;
  message: string;
  status: StudentFaceStatus;
};
