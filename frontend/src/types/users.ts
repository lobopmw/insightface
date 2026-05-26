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
