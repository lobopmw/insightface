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
