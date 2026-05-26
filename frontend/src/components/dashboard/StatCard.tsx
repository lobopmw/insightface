import type { ComponentType, ReactNode } from "react";

import { cn } from "@/lib/cn";

type StatCardProps = {
  label: string;
  value: ReactNode;
  icon: ComponentType<{ className?: string }>;
  helper?: string;
  tone?: "blue" | "green" | "amber" | "slate";
};

const tones = {
  blue: "bg-blue-50 text-blue-700 ring-blue-100",
  green: "bg-emerald-50 text-emerald-700 ring-emerald-100",
  amber: "bg-amber-50 text-amber-700 ring-amber-100",
  slate: "bg-slate-100 text-slate-700 ring-slate-200",
};

export function StatCard({ label, value, icon: Icon, helper, tone = "blue" }: StatCardProps) {
  return (
    <article className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-muted-foreground">{label}</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{value}</p>
        </div>
        <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg ring-1", tones[tone])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      {helper ? <p className="mt-3 text-sm text-muted-foreground">{helper}</p> : null}
    </article>
  );
}
