import { Clock3, Download, FileText, Filter, TimerReset } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { StatCard } from "@/components/dashboard/StatCard";
import { useBehaviorEpisodes, useReportSummary } from "@/hooks/useReportSummary";

export function Reports() {
  const { data, isLoading, isError } = useReportSummary();
  const { data: episodesData, isLoading: isLoadingEpisodes } = useBehaviorEpisodes();
  const items = data?.items ?? [];
  const episodes = episodesData?.items ?? [];
  const totalRecords = items.reduce((sum, item) => sum + item.records, 0);
  const totalSeconds = items.reduce((sum, item) => sum + item.duration_seconds, 0);

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-primary">Análise pedagógica</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Relatórios</h2>
          <p className="mt-1 text-sm text-muted-foreground">Resumo agregado a partir dos episódios comportamentais.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="inline-flex h-10 items-center gap-2 rounded-md border border-border bg-background px-3 text-sm font-medium text-slate-700 transition hover:bg-muted" type="button">
            <Filter className="h-4 w-4" />
            Filtros
          </button>
          <button className="inline-flex h-10 items-center gap-2 rounded-md bg-slate-950 px-3 text-sm font-medium text-white transition hover:bg-slate-800" type="button">
            <Download className="h-4 w-4" />
            Exportar
          </button>
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        <StatCard icon={FileText} label="Episodios" tone="blue" value={isLoading ? "..." : totalRecords} />
        <StatCard icon={TimerReset} label="Tempo observado" tone="green" value={isLoading ? "..." : `${Math.round(totalSeconds / 60)} min`} />
        <StatCard icon={Clock3} label="Registros recentes" tone="slate" value={isLoadingEpisodes ? "..." : episodes.length} />
      </section>

      <section className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-muted-foreground">Indicadores consolidados</p>
            <h2 className="mt-1 text-base font-semibold">Distribuição por comportamento</h2>
          </div>
          <span className="w-fit rounded-md bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
            {items.length} categorias
          </span>
        </div>
        {isLoading ? <p className="mt-4 text-sm text-muted-foreground">Carregando relatório...</p> : null}
        {isError ? <p className="mt-4 text-sm text-red-700">Não foi possível carregar o relatório.</p> : null}
        {!isLoading && !isError && items.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">Nenhum episódio encontrado para o escopo atual.</p>
        ) : null}

        {items.length > 0 ? (
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={items}>
                <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
                <XAxis dataKey="behavior" axisLine={false} tickLine={false} />
                <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "hsl(var(--muted) / 0.55)" }} />
                <Bar dataKey="records" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </section>

      <section className="overflow-hidden rounded-lg border border-border bg-card shadow-sm shadow-slate-200/60">
        <div className="border-b border-border px-4 py-4">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Auditoria</p>
          <h2 className="mt-1 text-base font-semibold">Episódios recentes</h2>
          <p className="text-sm text-muted-foreground">
            {isLoadingEpisodes ? "Carregando..." : `${episodesData?.total ?? 0} registros encontrados`}
          </p>
        </div>
        <div className="hidden grid-cols-[1.2fr_1fr_1fr_1fr_0.8fr] border-b border-border bg-muted/40 px-4 py-3 text-xs font-semibold uppercase text-muted-foreground md:grid">
          <span>Aluno</span>
          <span>Comportamento</span>
          <span>Disciplina</span>
          <span>Turma</span>
          <span>Duração</span>
        </div>
        {episodes.map((episode) => (
          <div
            key={episode.id}
            className="grid gap-2 border-b border-border px-4 py-3 text-sm transition hover:bg-muted/35 last:border-b-0 md:grid-cols-[1.2fr_1fr_1fr_1fr_0.8fr] md:gap-3"
          >
            <span className="font-medium">{episode.student}</span>
            <span className="text-muted-foreground">{episode.behavior}</span>
            <span className="text-muted-foreground">{episode.discipline ?? "-"}</span>
            <span className="text-muted-foreground">
              {[episode.class_name, episode.class_identifier].filter(Boolean).join(" ") || "-"}
            </span>
            <span className="text-muted-foreground">{Math.round(episode.duration_seconds)}s</span>
          </div>
        ))}
      </section>
    </div>
  );
}
