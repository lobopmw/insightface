import type { ComponentType, ReactNode } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";

import { cn } from "@/lib/cn";

type StatCardProps = {
  label: string;
  value: ReactNode;
  icon: ComponentType<{ className?: string }>;
  helper?: string;
  tone?: "blue" | "green" | "amber" | "slate";
  trend?: string;
  trendDirection?: "up" | "down" | "neutral";
};

const tones = {
  blue: "bg-cyan-50 text-cyan-700 ring-cyan-100",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-100",
  amber: "bg-amber-50 text-amber-700 ring-amber-100",
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
};

const trendStyles = {
  up: "bg-emerald-50 text-emerald-700",
  down: "bg-rose-50 text-rose-700",
  neutral: "bg-slate-100 text-slate-600",
};

export function StatCard({ label, value, icon: Icon, helper, tone = "blue", trend, trendDirection = "neutral" }: StatCardProps) {
  const TrendIcon = trendDirection === "down" ? TrendingDown : TrendingUp;

  return (
    <article className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold uppercase text-muted-foreground">{label}</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">{value}</p>
        </div>
        <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg ring-1", tones[tone])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <div className="mt-4 flex min-h-6 items-center justify-between gap-3">
        {helper ? <p className="truncate text-sm text-muted-foreground">{helper}</p> : <span />}
        {trend ? (
          <span className={cn("inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold", trendStyles[trendDirection])}>
            <TrendIcon className="h-3.5 w-3.5" />
            {trend}
          </span>
        ) : null}
      </div>
    </article>
  );
}
