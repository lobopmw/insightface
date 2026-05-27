import { Cpu, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { StudentFaceStatus } from "@/types/student";

export function EmbeddingStatusCard({
  canGenerate,
  isLoading,
  message,
  onGenerate,
  status,
}: {
  canGenerate: boolean;
  isLoading?: boolean;
  message?: string | null;
  onGenerate: () => void;
  status?: StudentFaceStatus;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <Cpu className="h-4 w-4 text-primary" />
        <h3 className="text-base font-semibold">Etapa 3 — Embeddings</h3>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Gera o embedding médio com a lógica legada do ClassAI/ArcFace e sincroniza com a estrutura atual.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button disabled={!canGenerate || isLoading || status?.embeddings_generated} onClick={onGenerate}>
          <Sparkles className="h-4 w-4" />
          {isLoading ? "Gerando..." : "Gerar embeddings"}
        </Button>
        <span className={status?.embeddings_generated ? "text-sm font-semibold text-emerald-700" : "text-sm font-medium text-amber-700"}>
          {status?.embeddings_generated ? "Aluno pronto para reconhecimento" : "Aguardando embeddings"}
        </span>
      </div>
      {message ? <p className="mt-3 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">{message}</p> : null}
    </section>
  );
}
