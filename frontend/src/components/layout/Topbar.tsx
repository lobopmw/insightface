import { Bell, Menu, Search } from "lucide-react";

import { useAuthStore } from "@/stores/authStore";
import { useNavigationStore } from "@/stores/navigationStore";

const pageTitles = {
  dashboard: "Dashboard",
  monitoring: "Monitoramento",
  students: "Alunos",
  reports: "Relatorios",
  settings: "Configuracoes",
} as const;

export function Topbar({ onMenuClick }: { onMenuClick: () => void }) {
  const currentPage = useNavigationStore((state) => state.currentPage);
  const user = useAuthStore((state) => state.user);
  const initials = (user?.nome ?? "US")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-border bg-white/90 px-4 backdrop-blur lg:px-6">
      <button
        className="flex h-10 w-10 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted hover:text-foreground lg:hidden"
        onClick={onMenuClick}
        type="button"
      >
        <Menu className="h-5 w-5" />
      </button>

      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Painel</p>
        <h1 className="truncate text-lg font-semibold">{pageTitles[currentPage]}</h1>
      </div>

      <div className="ml-auto hidden h-10 min-w-[260px] items-center gap-2 rounded-md border border-border bg-background px-3 text-sm text-muted-foreground md:flex">
        <Search className="h-4 w-4" />
        <span>Buscar alunos, sessoes e relatorios</span>
      </div>

      <button className="flex h-10 w-10 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted hover:text-foreground" type="button">
        <Bell className="h-4 w-4" />
      </button>

      <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-sm font-semibold text-primary-foreground">
        {initials || "US"}
      </div>
    </header>
  );
}
