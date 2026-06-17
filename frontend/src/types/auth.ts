export type LoginCredentials = {
  cpf: string;
  password: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type CurrentUser = {
  id: number;
  cpf: string;
  nome: string;
  cidade?: string | null;
  estado?: string | null;
  role: string;
  ativo: boolean;
  teacher_id?: number | null;
};
