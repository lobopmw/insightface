import type { Student } from "@/types/student";

export type AdminSubject = {
  id: number;
  nome: string;
};

export type AdminClass = {
  id: number;
  nome: string;
  identificador?: string | null;
};

export type AdminTeacher = {
  id: number;
  user_id: number;
  nome: string;
  cpf: string;
  email?: string | null;
  ativo: boolean;
};

export type AdminAssignment = {
  id: number;
  teacher_id: number;
  teacher_name: string;
  subject_id: number;
  subject_name: string;
  class_id: number;
  class_name: string;
  class_identifier?: string | null;
};

export type AdminCatalogResponse = {
  subjects: AdminSubject[];
  classes: AdminClass[];
  teachers: AdminTeacher[];
  assignments: AdminAssignment[];
};

export type AdminStudentListResponse = {
  items: Student[];
};
