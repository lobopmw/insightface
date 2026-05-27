import { CheckCircle2, Circle } from "lucide-react";

import type { PoseKey, StudentFaceStatus } from "@/types/student";

export const POSE_LABELS: Record<PoseKey, string> = {
  frontal: "Frontal",
  lateral_esquerda: "Lateral esquerda",
  lateral_direita: "Lateral direita",
  cabeca_baixa: "Cabeça baixa",
};

export const POSE_ORDER: PoseKey[] = ["frontal", "lateral_esquerda", "lateral_direita", "cabeca_baixa"];

export function PoseProgressCard({
  activePose,
  status,
  targetCount = 10,
}: {
  activePose: PoseKey;
  status?: StudentFaceStatus;
  targetCount?: number;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-3 shadow-sm">
      <h3 className="text-sm font-semibold">Progresso por pose</h3>
      <div className="mt-3 space-y-2.5">
        {POSE_ORDER.map((pose) => {
          const count = status?.[pose]?.count ?? 0;
          const complete = status?.[pose]?.complete ?? false;
          const percent = Math.min(100, Math.round((count / targetCount) * 100));
          const Icon = complete ? CheckCircle2 : Circle;
          return (
            <div key={pose} className="space-y-2">
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className={pose === activePose ? "font-semibold text-primary" : "font-medium"}>
                  {POSE_LABELS[pose]}
                </span>
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Icon className="h-4 w-4" />
                  {count}/{targetCount}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${percent}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
