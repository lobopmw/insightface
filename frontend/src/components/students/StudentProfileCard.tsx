import { BadgeCheck, UserRound } from "lucide-react";

import type { Student, StudentFaceStatus } from "@/types/student";

export function StudentProfileCard({ status, student }: { status?: StudentFaceStatus; student?: Student | null }) {
  return (
    <section className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <UserRound className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <h3 className="truncate text-base font-semibold">{student?.name ?? "Nenhum aluno selecionado"}</h3>
          <p className="text-sm text-muted-foreground">
            {[student?.matricula, student?.class_name, student?.class_identifier].filter(Boolean).join(" · ") || "Selecione ou cadastre um aluno"}
          </p>
        </div>
      </div>
      <div className="mt-4 rounded-md border border-border bg-background p-3 text-sm">
        <div className="flex items-center justify-between gap-3">
          <span className="text-muted-foreground">Reconhecimento</span>
          <span className={status?.embeddings_generated ? "font-semibold text-emerald-700" : "font-semibold text-amber-700"}>
            {status?.embeddings_generated ? "Pronto" : "Pendente"}
          </span>
        </div>
        {status?.embeddings_generated ? (
          <p className="mt-2 flex items-center gap-2 text-xs text-emerald-700">
            <BadgeCheck className="h-4 w-4" />
            Aluno pronto para reconhecimento.
          </p>
        ) : null}
      </div>
    </section>
  );
}
