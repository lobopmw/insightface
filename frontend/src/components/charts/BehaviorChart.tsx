import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts";

const data = [
  { behavior: "Atento", count: 0 },
  { behavior: "Distraído", count: 0 },
  { behavior: "Perguntando", count: 0 },
  { behavior: "Agitado", count: 0 },
  { behavior: "Dormindo", count: 0 },
];

export function BehaviorChart() {
  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="mb-4">
        <h2 className="text-base font-semibold">Comportamentos</h2>
        <p className="text-sm text-muted-foreground">Distribuição inicial aguardando integração.</p>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="behavior" />
            <YAxis allowDecimals={false} />
            <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
