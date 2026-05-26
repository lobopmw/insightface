import { Server, UserRound } from "lucide-react";

import { useAdminUsers } from "@/hooks/useAdminUsers";
import { useHealth } from "@/hooks/useHealth";
import { getApiBaseUrl } from "@/services/api";
import { getWebSocketBaseUrl } from "@/services/websocket";
import { useAuthStore } from "@/stores/authStore";

export function Settings() {
  const user = useAuthStore((state) => state.user);
  const { data: health, isError, isLoading } = useHealth();
  const { data: usersData } = useAdminUsers();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Configurações</h1>
        <p className="text-sm text-muted-foreground">Estado da nova stack e informações da sessão.</p>
      </header>

      <section className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <UserRound className="h-4 w-4 text-primary" />
            <h2 className="text-base font-semibold">Usuário</h2>
          </div>
          <dl className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Nome</dt>
              <dd className="font-medium">{user?.nome ?? "-"}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Perfil</dt>
              <dd className="font-medium">{user?.role ?? "-"}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">CPF</dt>
              <dd className="font-medium">{user?.cpf ?? "-"}</dd>
            </div>
          </dl>
        </article>

        <article className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-primary" />
            <h2 className="text-base font-semibold">Backend</h2>
          </div>
          <dl className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Status</dt>
              <dd className="font-medium">{isLoading ? "Verificando" : isError ? "Indisponível" : health?.status}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">API</dt>
              <dd className="break-all text-right font-medium">{getApiBaseUrl()}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">WebSocket</dt>
              <dd className="break-all text-right font-medium">{getWebSocketBaseUrl()}</dd>
            </div>
          </dl>
        </article>
      </section>

      {user?.role === "admin" ? (
        <section className="overflow-hidden rounded-lg border border-border bg-card">
          <div className="border-b border-border px-4 py-3">
            <h2 className="text-base font-semibold">Usuários</h2>
            <p className="text-sm text-muted-foreground">Controle administrativo inicial migrado para a API.</p>
          </div>
          <div className="grid grid-cols-[1.3fr_1fr_0.8fr_0.7fr] border-b border-border px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">
            <span>Nome</span>
            <span>CPF</span>
            <span>Perfil</span>
            <span>Status</span>
          </div>
          {(usersData?.items ?? []).map((adminUser) => (
            <div
              key={adminUser.id}
              className="grid grid-cols-[1.3fr_1fr_0.8fr_0.7fr] gap-3 border-b border-border px-4 py-3 text-sm last:border-b-0"
            >
              <span className="font-medium">{adminUser.nome}</span>
              <span className="text-muted-foreground">{adminUser.cpf}</span>
              <span className="text-muted-foreground">{adminUser.role}</span>
              <span className="text-muted-foreground">{adminUser.ativo ? "Ativo" : "Inativo"}</span>
            </div>
          ))}
        </section>
      ) : null}
    </div>
  );
}
