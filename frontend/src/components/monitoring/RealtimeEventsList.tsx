import type { MonitoringRealtimeEvent } from "@/types/monitoring";

function formatEventTime(timestamp?: string) {
  if (!timestamp) {
    return "--:--";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "--:--";
  }
  return date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function RealtimeEventsList({ events }: { events: MonitoringRealtimeEvent[] }) {
  return (
    <section className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold">Eventos em tempo real</h3>
        <span className="rounded-md bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">{events.length}</span>
      </div>
      <div className="mt-4 max-h-[420px] space-y-3 overflow-y-auto pr-1">
        {events.length === 0 ? (
          <p className="rounded-md border border-border bg-background p-3 text-sm text-muted-foreground">
            Aguardando eventos do WebSocket.
          </p>
        ) : null}
        {events.map((event, index) => (
          <div key={`${event.type}-${event.timestamp ?? index}`} className="rounded-md border border-border bg-background p-3">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="font-semibold text-foreground">{event.type}</span>
              <span className="text-muted-foreground">{formatEventTime(event.timestamp)}</span>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {event.student_name ?? "Aguardando identificação"} · {event.behavior ?? "Aguardando"} ·{" "}
              {Math.round((event.confidence ?? 0) * 100)}%
            </p>
            {event.camera_status ? <p className="mt-1 text-xs text-muted-foreground">Camera: {event.camera_status}</p> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
