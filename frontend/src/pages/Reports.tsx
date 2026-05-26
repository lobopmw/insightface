import { Clock3, FileText, TimerReset } from "lucide-react";
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
      <header className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <p className="text-sm font-medium text-primary">Analise pedagogica</p>
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">Relatorios</h2>
        <p className="mt-1 text-sm text-muted-foreground">Resumo agregado a partir dos episodios comportamentais.</p>
      </header>

      <section className="grid gap-4 md:grid-cols-3">
        <StatCard icon={FileText} label="Episodios" tone="blue" value={isLoading ? "..." : totalRecords} />
        <StatCard icon={TimerReset} label="Tempo observado" tone="green" value={isLoading ? "..." : `${Math.round(totalSeconds / 60)} min`} />
        <StatCard icon={Clock3} label="Registros recentes" tone="slate" value={isLoadingEpisodes ? "..." : episodes.length} />
      </section>

      <section className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <h2 className="text-base font-semibold">Distribuição por comportamento</h2>
        {isLoading ? <p className="mt-4 text-sm text-muted-foreground">Carregando relatório...</p> : null}
        {isError ? <p className="mt-4 text-sm text-red-700">Não foi possível carregar o relatório.</p> : null}
        {!isLoading && !isError && items.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">Nenhum episódio encontrado para o escopo atual.</p>
        ) : null}

        {items.length > 0 ? (
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={items}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="behavior" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="records" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </section>

      <section className="overflow-hidden rounded-lg border border-border bg-card">
        <div className="border-b border-border px-4 py-4">
          <h2 className="text-base font-semibold">Episódios recentes</h2>
          <p className="text-sm text-muted-foreground">
            {isLoadingEpisodes ? "Carregando..." : `${episodesData?.total ?? 0} registros encontrados`}
          </p>
        </div>
        <div className="grid grid-cols-[1.2fr_1fr_1fr_1fr_0.8fr] border-b border-border bg-muted/40 px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">
          <span>Aluno</span>
          <span>Comportamento</span>
          <span>Disciplina</span>
          <span>Turma</span>
          <span>Duração</span>
        </div>
        {episodes.map((episode) => (
          <div
            key={episode.id}
            className="grid grid-cols-[1.2fr_1fr_1fr_1fr_0.8fr] gap-3 border-b border-border px-4 py-3 text-sm transition hover:bg-muted/35 last:border-b-0"
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
