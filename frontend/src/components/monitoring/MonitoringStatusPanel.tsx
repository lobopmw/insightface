import { useEffect, useMemo, useState } from "react";
import { Activity, Timer, UserCheck } from "lucide-react";

import type { MonitoringStatus } from "@/types/monitoring";

type MonitoringStatusPanelProps = {
  isActive: boolean;
  status: MonitoringStatus;
};

function formatDuration(totalSeconds: number) {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const hours = String(Math.floor(safeSeconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((safeSeconds % 3600) / 60)).padStart(2, "0");
  const seconds = String(safeSeconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

export function MonitoringStatusPanel({ isActive, status }: MonitoringStatusPanelProps) {
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (isActive && startedAt === null) {
      setStartedAt(Date.now());
    }
    if (!isActive) {
      setStartedAt(null);
    }
  }, [isActive, startedAt]);

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [isActive]);

  const duration = useMemo(() => {
    if (!isActive || startedAt === null) {
      return "00:00:00";
    }
    return formatDuration((now - startedAt) / 1000);
  }, [isActive, now, startedAt]);

  const items = [
    { label: "Status", value: isActive ? "Em andamento" : "Aguardando início", icon: Activity, tone: "bg-emerald-50 text-emerald-700" },
    { label: "Duração da sessão", value: duration, icon: Timer, tone: "bg-cyan-50 text-cyan-700" },
    { label: "Alunos reconhecidos agora", value: String(status.recognized_students_now ?? 0), icon: UserCheck, tone: "bg-amber-50 text-amber-700" },
  ];

  return (
    <section className="grid gap-4 md:grid-cols-3">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <article key={item.label} className="rounded-lg border border-border bg-card p-4 shadow-sm shadow-slate-200/60">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase text-muted-foreground">{item.label}</p>
                <p className="mt-3 text-xl font-semibold tracking-tight">{item.value}</p>
              </div>
              <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${item.tone}`}>
                <Icon className="h-5 w-5" />
              </div>
            </div>
          </article>
        );
      })}
    </section>
  );
}
