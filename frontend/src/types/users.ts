export type AdminUser = {
  id: number;
  nome: string;
  cpf: string;
  cidade?: string | null;
  estado?: string | null;
  email?: string | null;
  role: string;
  ativo: boolean;
};

export type AdminUserListResponse = {
  items: AdminUser[];
};

export type AdminUserCreatePayload = {
  cpf: string;
  nome: string;
  password: string;
  cidade?: string | null;
  estado?: string | null;
  email?: string | null;
  role: string;
};

export type AdminUserUpdatePayload = {
  nome?: string | null;
  cidade?: string | null;
  estado?: string | null;
  email?: string | null;
  role?: string | null;
  ativo?: boolean | null;
};
