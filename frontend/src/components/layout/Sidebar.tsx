import {
  BarChart3,
  Camera,
  FileText,
  GraduationCap,
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

const navigationItems: Array<{ id: AppPage; label: string; icon: ComponentType<{ className?: string }> }> = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "monitoring", label: "Monitoramento", icon: Camera },
  { id: "students", label: "Alunos", icon: Users },
  { id: "reports", label: "Relatorios", icon: FileText },
  { id: "settings", label: "Configuracoes", icon: Settings },
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
          "fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col border-r border-border bg-white transition-transform duration-200 lg:sticky lg:top-0 lg:z-auto lg:h-screen lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
              <GraduationCap className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">InsightFace Classroom</p>
              <p className="truncate text-xs text-muted-foreground">{user?.nome ?? "Painel operacional"}</p>
            </div>
          </div>
          <button
            className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground lg:hidden"
            onClick={onClose}
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-3">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                className={cn(
                  "flex h-11 w-full items-center gap-3 rounded-md px-3 text-left text-sm font-medium transition",
                  isActive
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
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

        <div className="border-t border-border p-3">
          <div className="mb-3 rounded-lg border border-border bg-muted/40 p-3">
            <p className="truncate text-sm font-medium">{user?.nome ?? "Usuario"}</p>
            <p className="text-xs capitalize text-muted-foreground">{user?.role ?? "perfil"}</p>
          </div>
          <Button className="w-full justify-start" variant="ghost" onClick={logout}>
            <LogOut className="h-4 w-4" />
            Sair
          </Button>
        </div>
      </aside>
    </>
  );
}
