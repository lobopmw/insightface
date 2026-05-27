import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const data = [
  { behavior: "Atento", count: 0 },
  { behavior: "Distraído", count: 0 },
  { behavior: "Perguntando", count: 0 },
  { behavior: "Agitado", count: 0 },
  { behavior: "Dormindo", count: 0 },
];

export function BehaviorChart() {
  return (
    <section className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-muted-foreground">Distribuição comportamental</p>
          <h2 className="mt-1 text-base font-semibold">Leitura da turma</h2>
        </div>
        <span className="inline-flex w-fit items-center rounded-md bg-cyan-50 px-2.5 py-1 text-xs font-semibold text-cyan-700">
          tempo real
        </span>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
            <XAxis dataKey="behavior" axisLine={false} tickLine={false} />
            <YAxis allowDecimals={false} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "hsl(var(--muted) / 0.55)" }} />
            <Bar dataKey="count" fill="hsl(var(--primary))" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
