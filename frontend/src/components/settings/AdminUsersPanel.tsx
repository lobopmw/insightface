import { Edit3, Plus, Save, ShieldCheck, Trash2, UserRound, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/Button";
import { createAdminUser, updateAdminUser } from "@/features/auth/usersApi";
import { useAuthStore } from "@/stores/authStore";
import type { AdminUser, AdminUserCreatePayload, AdminUserUpdatePayload } from "@/types/users";

type AdminUsersPanelProps = {
  users: AdminUser[];
};

const emptyCreateForm: AdminUserCreatePayload = {
  cpf: "",
  nome: "",
  password: "",
  cidade: "",
  estado: "",
  email: "",
  role: "professor",
};

function normalizeUserForm(payload: AdminUserCreatePayload): AdminUserCreatePayload {
  return {
    cpf: payload.cpf.replace(/\D/g, ""),
    nome: payload.nome.trim(),
    password: payload.password,
    cidade: payload.cidade?.trim() || null,
    estado: payload.estado?.trim().toUpperCase() || null,
    email: payload.email?.trim() || null,
    role: payload.role,
  };
}

function buildUpdatePayload(user: AdminUser): AdminUserUpdatePayload {
  return {
    nome: user.nome.trim(),
    cidade: user.cidade?.trim() || null,
    estado: user.estado?.trim().toUpperCase() || null,
    email: user.email?.trim() || null,
    role: user.role,
    ativo: user.ativo,
  };
}

export function AdminUsersPanel({ users }: AdminUsersPanelProps) {
  const token = useAuthStore((state) => state.token);
  const currentUser = useAuthStore((state) => state.user);
  const queryClient = useQueryClient();
  const [createForm, setCreateForm] = useState<AdminUserCreatePayload>(emptyCreateForm);
  const [editingUserId, setEditingUserId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<AdminUser | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const activeUsers = useMemo(() => users.filter((user) => user.ativo).length, [users]);

  const createMutation = useMutation({
    mutationFn: (payload: AdminUserCreatePayload) => createAdminUser(token as string, normalizeUserForm(payload)),
    onSuccess: () => {
      setCreateForm(emptyCreateForm);
      setMessage("Usuário cadastrado com sucesso.");
      void queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      void queryClient.invalidateQueries({ queryKey: ["admin-catalog"] });
    },
    onError: () => setMessage("Não foi possível cadastrar o usuário. Verifique CPF, senha e permissões."),
  });

  const updateMutation = useMutation({
    mutationFn: ({ userId, payload }: { userId: number; payload: AdminUserUpdatePayload }) =>
      updateAdminUser(token as string, userId, payload),
    onSuccess: () => {
      setEditingUserId(null);
      setEditForm(null);
      setMessage("Usuário atualizado com sucesso.");
      void queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      void queryClient.invalidateQueries({ queryKey: ["admin-catalog"] });
    },
    onError: () => setMessage("Não foi possível atualizar o usuário."),
  });

  const canCreate = createForm.nome.trim().length > 0 && createForm.cpf.replace(/\D/g, "").length === 11 && createForm.password.length >= 6;

  return (
    <section className="overflow-hidden rounded-lg border border-border bg-card shadow-sm shadow-slate-200/60">
      <div className="border-b border-border px-4 py-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-muted-foreground">Área admin</p>
            <h2 className="mt-1 text-base font-semibold">Gerenciamento de usuários</h2>
            <p className="text-sm text-muted-foreground">Cadastre, edite e desative acessos do sistema.</p>
          </div>
          <div className="flex gap-2 text-xs font-semibold">
            <span className="rounded-md bg-cyan-50 px-2.5 py-1 text-cyan-700">{users.length} cadastrados</span>
            <span className="rounded-md bg-emerald-50 px-2.5 py-1 text-emerald-700">{activeUsers} ativos</span>
          </div>
        </div>
      </div>

      <form
        className="grid gap-3 border-b border-border bg-slate-50/80 p-4 lg:grid-cols-[1.15fr_0.85fr_0.8fr_0.8fr_0.7fr_auto]"
        onSubmit={(event) => {
          event.preventDefault();
          if (canCreate) {
            createMutation.mutate(createForm);
          }
        }}
      >
        <input
          className="h-10 rounded-md border border-border bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          onChange={(event) => setCreateForm((current) => ({ ...current, nome: event.target.value }))}
          placeholder="Nome completo"
          value={createForm.nome}
        />
        <input
          className="h-10 rounded-md border border-border bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          maxLength={11}
          onChange={(event) => setCreateForm((current) => ({ ...current, cpf: event.target.value.replace(/\D/g, "") }))}
          placeholder="CPF"
          value={createForm.cpf}
        />
        <input
          className="h-10 rounded-md border border-border bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          onChange={(event) => setCreateForm((current) => ({ ...current, password: event.target.value }))}
          placeholder="Senha inicial"
          type="password"
          value={createForm.password}
        />
        <input
          className="h-10 rounded-md border border-border bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          onChange={(event) => setCreateForm((current) => ({ ...current, email: event.target.value }))}
          placeholder="E-mail"
          type="email"
          value={createForm.email ?? ""}
        />
        <select
          className="h-10 rounded-md border border-border bg-white px-3 text-sm"
          onChange={(event) => setCreateForm((current) => ({ ...current, role: event.target.value }))}
          value={createForm.role}
        >
          <option value="professor">Professor</option>
          <option value="admin">Admin</option>
        </select>
        <Button disabled={!canCreate || createMutation.isPending} type="submit">
          <Plus className="h-4 w-4" />
          Cadastrar
        </Button>
      </form>

      {message ? <p className="border-b border-border px-4 py-3 text-sm text-muted-foreground">{message}</p> : null}

      <div className="hidden grid-cols-[1.2fr_0.9fr_0.9fr_0.7fr_0.7fr_148px] border-b border-border bg-muted/40 px-4 py-3 text-xs font-semibold uppercase text-muted-foreground lg:grid">
        <span>Nome</span>
        <span>CPF</span>
        <span>E-mail</span>
        <span>Perfil</span>
        <span>Status</span>
        <span>Ações</span>
      </div>

      {(users ?? []).map((adminUser) => {
        const isEditing = editingUserId === adminUser.id && editForm;
        const isCurrentUser = currentUser?.id === adminUser.id;

        return (
          <div
            key={adminUser.id}
            className="grid gap-3 border-b border-border px-4 py-3 text-sm transition hover:bg-muted/35 last:border-b-0 lg:grid-cols-[1.2fr_0.9fr_0.9fr_0.7fr_0.7fr_148px] lg:items-center"
          >
            {isEditing ? (
              <>
                <input
                  className="h-9 rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
                  onChange={(event) => setEditForm((current) => (current ? { ...current, nome: event.target.value } : current))}
                  value={editForm.nome}
                />
                <span className="text-muted-foreground">{adminUser.cpf}</span>
                <input
                  className="h-9 rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
                  onChange={(event) => setEditForm((current) => (current ? { ...current, email: event.target.value } : current))}
                  value={editForm.email ?? ""}
                />
                <select
                  className="h-9 rounded-md border border-border bg-background px-3"
                  onChange={(event) => setEditForm((current) => (current ? { ...current, role: event.target.value } : current))}
                  value={editForm.role}
                >
                  <option value="professor">Professor</option>
                  <option value="admin">Admin</option>
                </select>
                <select
                  className="h-9 rounded-md border border-border bg-background px-3"
                  disabled={isCurrentUser}
                  onChange={(event) => setEditForm((current) => (current ? { ...current, ativo: event.target.value === "true" } : current))}
                  value={String(editForm.ativo)}
                >
                  <option value="true">Ativo</option>
                  <option value="false">Inativo</option>
                </select>
              </>
            ) : (
              <>
                <span className="flex items-center gap-2 font-medium">
                  {adminUser.role === "admin" ? <ShieldCheck className="h-4 w-4 text-cyan-700" /> : <UserRound className="h-4 w-4 text-muted-foreground" />}
                  {adminUser.nome}
                </span>
                <span className="text-muted-foreground">{adminUser.cpf}</span>
                <span className="truncate text-muted-foreground">{adminUser.email ?? "-"}</span>
                <span className="capitalize text-muted-foreground">{adminUser.role}</span>
                <span className={adminUser.ativo ? "font-medium text-emerald-700" : "font-medium text-rose-700"}>
                  {adminUser.ativo ? "Ativo" : "Inativo"}
                </span>
              </>
            )}

            <div className="flex flex-wrap gap-2 lg:justify-end">
              {isEditing ? (
                <>
                  <button
                    className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground transition hover:bg-muted hover:text-foreground"
                    onClick={() => {
                      setEditingUserId(null);
                      setEditForm(null);
                    }}
                    title="Cancelar edição"
                    type="button"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <button
                    className="flex h-9 w-9 items-center justify-center rounded-md bg-slate-950 text-white transition hover:bg-slate-800 disabled:opacity-50"
                    disabled={updateMutation.isPending || !editForm.nome.trim()}
                    onClick={() => updateMutation.mutate({ userId: adminUser.id, payload: buildUpdatePayload(editForm) })}
                    title="Salvar alterações"
                    type="button"
                  >
                    <Save className="h-4 w-4" />
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground transition hover:bg-muted hover:text-foreground"
                    onClick={() => {
                      setEditingUserId(adminUser.id);
                      setEditForm(adminUser);
                      setMessage(null);
                    }}
                    title="Editar usuário"
                    type="button"
                  >
                    <Edit3 className="h-4 w-4" />
                  </button>
                  <button
                    className="flex h-9 w-9 items-center justify-center rounded-md border border-rose-200 text-rose-700 transition hover:bg-rose-50 disabled:opacity-50"
                    disabled={isCurrentUser || updateMutation.isPending || !adminUser.ativo}
                    onClick={() => updateMutation.mutate({ userId: adminUser.id, payload: { ativo: false } })}
                    title={isCurrentUser ? "Não é possível desativar o usuário logado" : "Excluir/desativar usuário"}
                    type="button"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </>
              )}
            </div>
          </div>
        );
      })}
    </section>
  );
}
