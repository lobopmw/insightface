import {
  BarChart3,
  Camera,
  FileText,
  GraduationCap,
  LogOut,
  Settings,
  Users,
} from "lucide-react";
import type { ComponentType, ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { useAuthStore } from "@/stores/authStore";
import { useNavigationStore } from "@/stores/navigationStore";

export type AppPage = "dashboard" | "monitoring" | "students" | "reports" | "settings";

const navigationItems: Array<{ id: AppPage; label: string; icon: ComponentType<{ className?: string }> }> = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "monitoring", label: "Monitoramento", icon: Camera },
  { id: "students", label: "Alunos", icon: Users },
  { id: "reports", label: "Relatórios", icon: FileText },
  { id: "settings", label: "Configurações", icon: Settings },
];

export function AppLayout({ children }: { children: ReactNode }) {
  const currentPage = useNavigationStore((state) => state.currentPage);
  const setCurrentPage = useNavigationStore((state) => state.setCurrentPage);
  const logout = useAuthStore((state) => state.logout);
  const user = useAuthStore((state) => state.user);

  return (
    <div className="min-h-screen bg-background text-foreground lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-border bg-card lg:min-h-screen lg:border-b-0 lg:border-r">
        <div className="flex h-16 items-center gap-3 border-b border-border px-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <GraduationCap className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-semibold">Monitoramento SEDUC</p>
            <p className="text-xs text-muted-foreground">{user?.nome ?? "Migração React + FastAPI"}</p>
          </div>
        </div>

        <nav className="flex gap-2 overflow-x-auto p-3 lg:block lg:space-y-1">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                className={`flex min-h-10 min-w-max items-center gap-3 rounded-md px-3 text-sm transition ${
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
                onClick={() => setCurrentPage(item.id)}
                type="button"
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="hidden border-t border-border p-3 lg:block">
          <Button className="w-full justify-start" variant="ghost" onClick={logout}>
            <LogOut className="h-4 w-4" />
            Sair
          </Button>
        </div>
      </aside>

      <main className="min-w-0 p-4 sm:p-6 lg:p-8">{children}</main>
    </div>
  );
}
