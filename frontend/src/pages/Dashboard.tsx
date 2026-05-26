import { Activity, AlertTriangle, Users } from "lucide-react";

import { BehaviorChart } from "@/components/charts/BehaviorChart";
import { useDashboardOverview } from "@/hooks/useDashboardOverview";

export function Dashboard() {
  const { data, isLoading } = useDashboardOverview();
  const stats = [
    { label: "Sessões ativas", value: data?.active_sessions ?? 0, icon: Activity },
    { label: "Alunos monitorados", value: data?.students_monitored ?? 0, icon: Users },
    { label: "Alertas pedagógicos", value: data?.alerts ?? 0, icon: AlertTriangle },
  ];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-muted-foreground">Visão inicial preparada para dados reais do backend.</p>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <article key={stat.label} className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">{stat.label}</span>
                <Icon className="h-4 w-4 text-primary" />
              </div>
              <p className="mt-3 text-3xl font-semibold">{isLoading ? "..." : stat.value}</p>
            </article>
          );
        })}
      </section>

      <BehaviorChart />
    </div>
  );
}
