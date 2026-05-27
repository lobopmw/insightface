import {
  BarChart3,
  Camera,
  CheckCircle2,
  FileText,
  LogOut,
  Settings,
  Users,
  X,
} from "lucide-react";
import type { ComponentType } from "react";

import { Button } from "@/components/ui/Button";
import type { AppPage } from "@/components/layout/AppLayout";
import { cn } from "@/lib/cn";
import { useAuthStore } from "@/stores/authStore";
import { useNavigationStore } from "@/stores/navigationStore";

const navigationItems: Array<{ id: AppPage; label: string; icon: ComponentType<{ className?: string }>; adminOnly?: boolean }> = [
  { id: "dashboard", label: "Visão geral", icon: BarChart3 },
  { id: "monitoring", label: "Monitoramento", icon: Camera },
  { id: "students", label: "Alunos", icon: Users, adminOnly: true },
  { id: "reports", label: "Relatórios", icon: FileText },
  { id: "settings", label: "Configurações", icon: Settings },
];

type SidebarProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const currentPage = useNavigationStore((state) => state.currentPage);
  const setCurrentPage = useNavigationStore((state) => state.setCurrentPage);
  const logout = useAuthStore((state) => state.logout);
  const user = useAuthStore((state) => state.user);
  const visibleNavigationItems = navigationItems.filter((item) => !item.adminOnly || user?.role === "admin");

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-slate-950/40 backdrop-blur-sm transition-opacity lg:hidden",
          isOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex w-[288px] flex-col border-r border-slate-800 bg-slate-950 text-white transition-transform duration-200 lg:sticky lg:top-0 lg:z-auto lg:h-screen lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-white/10 px-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-cyan-500 text-white shadow-sm shadow-cyan-950/40">
              <img alt="" className="h-7 w-7" src="/classai-icon.svg" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">ClassAI</p>
              <p className="truncate text-xs text-slate-400">Monitoramento escolar</p>
            </div>
          </div>
          <button
            className="flex h-9 w-9 items-center justify-center rounded-md text-slate-400 hover:bg-white/10 hover:text-white lg:hidden"
            onClick={onClose}
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="border-b border-white/10 px-4 py-4">
          <div className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs font-medium uppercase text-slate-400">Operação</span>
              <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-400/10 px-2 py-1 text-xs font-semibold text-emerald-300">
                <CheckCircle2 className="h-3.5 w-3.5" />
                Online
              </span>
            </div>
            <p className="mt-3 text-sm font-medium">{user?.nome ?? "Painel operacional"}</p>
            <p className="mt-1 text-xs capitalize text-slate-400">{user?.role ?? "perfil"} · análise em tempo real</p>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-3">
          {visibleNavigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                className={cn(
                  "flex h-11 w-full items-center gap-3 rounded-md px-3 text-left text-sm font-medium transition",
                  isActive
                    ? "bg-white text-slate-950 shadow-sm"
                    : "text-slate-400 hover:bg-white/10 hover:text-white",
                )}
                onClick={() => {
                  setCurrentPage(item.id);
                  onClose();
                }}
                type="button"
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-white/10 p-3">
          <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-md border border-white/10 bg-white/[0.04] p-2">
              <p className="text-slate-400">Sessões</p>
              <p className="mt-1 font-semibold">Tempo real</p>
            </div>
            <div className="rounded-md border border-white/10 bg-white/[0.04] p-2">
              <p className="text-slate-400">Relatórios</p>
              <p className="mt-1 font-semibold">Ativos</p>
            </div>
          </div>
          <Button className="w-full justify-start text-slate-300 hover:bg-white/10 hover:text-white" variant="ghost" onClick={logout}>
            <LogOut className="h-4 w-4" />
            Sair
          </Button>
        </div>
      </aside>
    </>
  );
}
