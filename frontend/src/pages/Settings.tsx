import { Database, Server, ShieldCheck, UserRound } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { StatCard } from "@/components/dashboard/StatCard";
import { AdminCatalogPanel, type AdminCatalogTab } from "@/components/settings/AdminCatalogPanel";
import { AdminUsersPanel } from "@/components/settings/AdminUsersPanel";
import { useAdminCatalog } from "@/hooks/useAdminCatalog";
import { useAdminUsers } from "@/hooks/useAdminUsers";
import { useHealth } from "@/hooks/useHealth";
import { getApiBaseUrl } from "@/services/api";
import { getWebSocketBaseUrl } from "@/services/websocket";
import { useAuthStore } from "@/stores/authStore";

export function Settings() {
  const user = useAuthStore((state) => state.user);
  const { data: health, isError, isLoading } = useHealth();
  const { data: usersData } = useAdminUsers();
  const { data: catalogData } = useAdminCatalog();
  const [activeAdminStep, setActiveAdminStep] = useState<AdminStep>("system");
  const isAdmin = user?.role === "admin";
  const adminUsers = usersData?.items ?? [];
  const hasProfessor = adminUsers.some((adminUser) => adminUser.role === "professor" && adminUser.ativo);
  const hasSubjects = Boolean(catalogData?.subjects.length);
  const hasClasses = Boolean(catalogData?.classes.length);
  const hasAssignments = Boolean(catalogData?.assignments.length);

  const adminSteps = useMemo(
    () => [
      { id: "system" as const, label: "Sistema", enabled: true },
      { id: "users" as const, label: "Usuários", enabled: true },
      { id: "subjects" as const, label: "Disciplinas", enabled: hasProfessor },
      { id: "classes" as const, label: "Classes", enabled: hasProfessor && hasSubjects },
      { id: "assignments" as const, label: "Vínculos", enabled: hasProfessor && hasSubjects && hasClasses },
      { id: "students" as const, label: "Alunos", enabled: hasAssignments || hasClasses },
    ],
    [hasAssignments, hasClasses, hasProfessor, hasSubjects],
  );

  useEffect(() => {
    if (!isAdmin) {
      setActiveAdminStep("system");
      return;
    }
    const currentStep = adminSteps.find((step) => step.id === activeAdminStep);
    if (currentStep && !currentStep.enabled) {
      setActiveAdminStep("system");
    }
  }, [activeAdminStep, adminSteps, isAdmin]);

  return (
    <div className="space-y-6">
      <header className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <p className="text-sm font-medium text-primary">Administracao</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">Configuracoes</h2>
        <p className="mt-1 text-sm text-muted-foreground">Estado da nova stack e informacoes da sessao.</p>
      </header>

      {isAdmin ? (
        <section className="rounded-lg border border-border bg-card p-3 shadow-sm shadow-slate-200/60">
          <div className="flex gap-2 overflow-x-auto">
            {adminSteps.map((step, index) => (
              <button
                key={step.id}
                className={`flex h-10 shrink-0 items-center gap-2 rounded-md px-3 text-sm font-semibold transition ${
                  activeAdminStep === step.id
                    ? "bg-slate-950 text-white"
                    : step.enabled
                      ? "bg-background text-muted-foreground hover:bg-muted hover:text-foreground"
                      : "cursor-not-allowed bg-slate-100 text-slate-400"
                }`}
                disabled={!step.enabled}
                onClick={() => setActiveAdminStep(step.id)}
                type="button"
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-sm bg-white/15 text-xs">{index + 1}</span>
                {step.label}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {activeAdminStep === "system" ? (
        <>
          <section className="grid gap-4 md:grid-cols-3">
            <StatCard icon={ShieldCheck} label="Perfil" tone="blue" value={user?.role ?? "-"} />
            <StatCard icon={Server} label="Backend" tone={isError ? "amber" : "green"} value={isLoading ? "..." : isError ? "Offline" : "Online"} />
            <StatCard icon={Database} label="Usuarios" tone="slate" value={usersData?.items?.length ?? "-"} />
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <article className="rounded-lg border border-border bg-card p-5 shadow-sm">
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

            <article className="rounded-lg border border-border bg-card p-5 shadow-sm">
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
        </>
      ) : null}

      {isAdmin && activeAdminStep === "users" ? <AdminUsersPanel users={adminUsers} /> : null}
      {isAdmin && isCatalogStep(activeAdminStep) ? <AdminCatalogPanel activeTab={activeAdminStep} /> : null}
    </div>
  );
}

type AdminStep = "system" | "users" | AdminCatalogTab;

function isCatalogStep(step: AdminStep): step is AdminCatalogTab {
  return step === "subjects" || step === "classes" || step === "assignments" || step === "students";
}
