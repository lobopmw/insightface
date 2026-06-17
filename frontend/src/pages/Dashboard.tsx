import {
  Activity,
  AlertTriangle,
  BarChart3,
  Clock3,
  FileText,
  LineChart as LineChartIcon,
  PlayCircle,
  TrendingUp,
  UserPlus,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/Button";
import { useDashboardOverview } from "@/hooks/useDashboardOverview";
import { cn } from "@/lib/cn";
import { useNavigationStore } from "@/stores/navigationStore";

const behaviorDistribution = [
  { behavior: "Atento", value: 0, color: "#10b981" },
  { behavior: "Distraído", value: 0, color: "#f59e0b" },
  { behavior: "Perguntando", value: 0, color: "#06b6d4" },
  { behavior: "Agitado", value: 0, color: "#ef4444" },
  { behavior: "Dormindo", value: 0, color: "#64748b" },
];

const timelineData = [
  { time: "08:00", attention: 0 },
  { time: "09:00", attention: 0 },
  { time: "10:00", attention: 0 },
  { time: "11:00", attention: 0 },
];

const highlightedStudents: Array<{ name: string; behavior: string; riskTime: string }> = [];

// TODO: substituir estes dados temporarios por um endpoint pedagogico consolidado
// com distribuicao comportamental, linha do tempo e destaques por aluno.
const pedagogicalSnapshot = {
  totalMonitoredToday: "0 min",
  classAttention: {
    label: "Sem dados",
    value: "Aguardando sessão",
    tone: "slate" as const,
  },
  predominantBehavior: "Sem dados hoje",
  summary: "A turma ainda não possui dados suficientes para análise pedagógica hoje.",
};

const attentionTones = {
  green: "border-emerald-200 bg-emerald-50 text-emerald-700",
  amber: "border-amber-200 bg-amber-50 text-amber-700",
  red: "border-rose-200 bg-rose-50 text-rose-700",
  slate: "border-slate-200 bg-slate-50 text-slate-600",
};

const statTones = {
  green: "bg-emerald-50 text-emerald-700 ring-emerald-100",
  blue: "bg-cyan-50 text-cyan-700 ring-cyan-100",
  amber: "bg-amber-50 text-amber-700 ring-amber-100",
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
};

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-40 items-center justify-center rounded-md border border-dashed border-border bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
      {message}
    </div>
  );
}

function MetricCard({
  helper,
  icon: Icon,
  label,
  tone,
  value,
}: {
  helper: string;
  icon: typeof Activity;
  label: string;
  tone: keyof typeof statTones;
  value: string | number;
}) {
  return (
    <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold uppercase text-muted-foreground">{label}</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">{value}</p>
        </div>
        <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg ring-1", statTones[tone])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <p className="mt-4 truncate text-sm text-muted-foreground">{helper}</p>
    </article>
  );
}

export function Dashboard() {
  const { data, isLoading } = useDashboardOverview();
  const setCurrentPage = useNavigationStore((state) => state.setCurrentPage);
  const hasBehaviorData = behaviorDistribution.some((item) => item.value > 0);
  const hasTimelineData = timelineData.some((item) => item.attention > 0);

  const stats = [
    {
      label: "Sessões ativas",
      value: data?.active_sessions ?? 0,
      icon: Activity,
      tone: "green" as const,
      helper: "turmas em acompanhamento agora",
    },
    {
      label: "Alunos monitorados",
      value: data?.students_monitored ?? 0,
      icon: Users,
      tone: "blue" as const,
      helper: "alunos reconhecidos no período",
    },
    {
      label: "Alertas pedagógicos",
      value: data?.alerts ?? 0,
      icon: AlertTriangle,
      tone: "amber" as const,
      helper: "situações que pedem atenção",
    },
    {
      label: "Tempo monitorado hoje",
      value: pedagogicalSnapshot.totalMonitoredToday,
      icon: Clock3,
      tone: "slate" as const,
      helper: "somatório das sessões concluídas",
    },
  ];

  const quickActions = [
    { label: "Iniciar monitoramento", icon: PlayCircle, page: "monitoring" as const, variant: "primary" as const },
    { label: "Cadastrar aluno", icon: UserPlus, page: "students" as const, variant: "secondary" as const },
    { label: "Gerar relatório", icon: FileText, page: "reports" as const, variant: "secondary" as const },
    { label: "Ver analytics", icon: BarChart3, page: "reports" as const, variant: "ghost" as const },
  ];

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-950">Visão geral</h1>
        <p className="text-sm text-muted-foreground">Acompanhamento pedagógico e operacional da plataforma</p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat) => (
          <MetricCard
            helper={stat.helper}
            icon={stat.icon}
            key={stat.label}
            label={stat.label}
            tone={stat.tone}
            value={isLoading ? "..." : stat.value}
          />
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Atenção da turma</p>
          <div className="mt-4 flex items-center justify-between gap-4">
            <div>
              <p className="text-2xl font-semibold text-slate-950">{pedagogicalSnapshot.classAttention.label}</p>
              <p className="mt-1 text-sm text-muted-foreground">{pedagogicalSnapshot.classAttention.value}</p>
            </div>
            <span
              className={cn(
                "rounded-md border px-2.5 py-1 text-xs font-semibold",
                attentionTones[pedagogicalSnapshot.classAttention.tone],
              )}
            >
              Hoje
            </span>
          </div>
        </article>

        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Comportamento predominante</p>
          <div className="mt-4 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-cyan-50 text-cyan-700 ring-1 ring-cyan-100">
              <TrendingUp className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xl font-semibold text-slate-950">{pedagogicalSnapshot.predominantBehavior}</p>
              <p className="mt-1 text-sm text-muted-foreground">com base no período atual</p>
            </div>
          </div>
        </article>

        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Resumo pedagógico</p>
          <p className="mt-4 text-sm leading-6 text-slate-700">{pedagogicalSnapshot.summary}</p>
        </article>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.35fr_0.65fr]">
        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase text-muted-foreground">Distribuição comportamental</p>
              <h2 className="mt-1 text-base font-semibold">Leitura do período</h2>
            </div>
            <span className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">Hoje</span>
          </div>
          {hasBehaviorData ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={behaviorDistribution}>
                  <CartesianGrid stroke="hsl(var(--border))" vertical={false} />
                  <XAxis axisLine={false} dataKey="behavior" tickLine={false} />
                  <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
                  <Tooltip cursor={{ fill: "hsl(var(--muted) / 0.55)" }} />
                  <Bar dataKey="value" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState message="Nenhuma sessão monitorada ainda hoje." />
          )}
        </article>

        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Alunos com maior necessidade de atenção</p>
          <div className="mt-4 space-y-3">
            {highlightedStudents.length > 0 ? (
              highlightedStudents.slice(0, 5).map((student) => (
                <div className="rounded-md border border-border bg-background p-3" key={student.name}>
                  <p className="font-medium text-slate-950">{student.name}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{student.behavior}</p>
                  <p className="mt-2 text-xs font-semibold text-amber-700">{student.riskTime}</p>
                </div>
              ))
            ) : (
              <EmptyState message="Nenhum aluno em destaque no momento." />
            )}
          </div>
        </article>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase text-muted-foreground">Linha temporal resumida</p>
              <h2 className="mt-1 text-base font-semibold">Evolução da atenção</h2>
            </div>
            <LineChartIcon className="h-5 w-5 text-muted-foreground" />
          </div>
          {hasTimelineData ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timelineData}>
                  <CartesianGrid stroke="hsl(var(--border))" vertical={false} />
                  <XAxis axisLine={false} dataKey="time" tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Line dataKey="attention" dot={false} stroke="hsl(var(--primary))" strokeWidth={3} type="monotone" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState message="A evolução por horário aparecerá após o início do monitoramento." />
          )}
        </article>

        <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Ações rápidas</p>
          <div className="mt-4 grid gap-2">
            {quickActions.map((action) => {
              const Icon = action.icon;
              return (
                <Button
                  className="h-10 justify-start"
                  key={action.label}
                  onClick={() => setCurrentPage(action.page)}
                  variant={action.variant}
                >
                  <Icon className="h-4 w-4" />
                  {action.label}
                </Button>
              );
            })}
          </div>
        </article>
      </section>
    </div>
  );
}
