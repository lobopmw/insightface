import { Activity, AlertTriangle, CalendarDays, Users } from "lucide-react";

import { BehaviorChart } from "@/components/charts/BehaviorChart";
import { StatCard } from "@/components/dashboard/StatCard";
import { useDashboardOverview } from "@/hooks/useDashboardOverview";

export function Dashboard() {
  const { data, isLoading } = useDashboardOverview();
  const stats = [
    { label: "Sessões ativas", value: data?.active_sessions ?? 0, icon: Activity, tone: "green" as const },
    { label: "Alunos monitorados", value: data?.students_monitored ?? 0, icon: Users, tone: "blue" as const },
    { label: "Alertas pedagógicos", value: data?.alerts ?? 0, icon: AlertTriangle, tone: "amber" as const },
  ];

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 rounded-lg border border-border bg-card p-5 shadow-sm lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-sm font-medium text-primary">Visao operacional</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">Resumo do monitoramento</h2>
          <p className="mt-1 text-sm text-muted-foreground">Indicadores consolidados da nova stack React + FastAPI.</p>
        </div>
        <div className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm text-muted-foreground">
          <CalendarDays className="h-4 w-4" />
          Hoje
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        {stats.map((stat) => (
          <StatCard
            key={stat.label}
            helper="Atualizado pela API"
            icon={stat.icon}
            label={stat.label}
            tone={stat.tone}
            value={isLoading ? "..." : stat.value}
          />
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
        <BehaviorChart />
        <article className="rounded-lg border border-border bg-card p-5 shadow-sm">
          <h3 className="text-base font-semibold">Fila operacional</h3>
          <div className="mt-4 space-y-3">
            {[
              ["Camera", "Aguardando conexao do runtime"],
              ["IA", "Modelos preservados no legado"],
              ["WebSocket", "Canal base disponivel"],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between gap-4 rounded-md border border-border bg-background px-3 py-3 text-sm">
                <span className="font-medium">{label}</span>
                <span className="text-right text-muted-foreground">{value}</span>
              </div>
            ))}
          </div>
        </article>
      </section>
    </div>
  );
}
