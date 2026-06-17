import { CheckCircle2, Clock3, ImageIcon, ScanFace, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { PoseKey, Student, StudentFaceStatus } from "@/types/student";

type StudentTableProps = {
  isLoading?: boolean;
  selectedStudentId?: string | null;
  statuses?: Record<string, StudentFaceStatus>;
  students: Student[];
  onDelete: (studentId: string) => void;
  onSelect: (student: Student) => void;
};

const poses: Array<{ key: PoseKey; label: string }> = [
  { key: "frontal", label: "Frontal" },
  { key: "lateral_esquerda", label: "Esq." },
  { key: "lateral_direita", label: "Dir." },
  { key: "cabeca_baixa", label: "Baixa" },
];

function getBiometrySummary(status?: StudentFaceStatus) {
  if (!status) {
    return {
      label: "Selecionar aluno",
      detail: "Clique em biometria para carregar",
      total: 0,
      icon: Clock3,
      className: "border-slate-200 bg-slate-50 text-slate-600",
    };
  }

  const total = poses.reduce((sum, pose) => sum + (status[pose.key]?.count ?? 0), 0);
  const complete = poses.every((pose) => status[pose.key]?.complete);

  if (status.embeddings_generated) {
    return {
      label: "Pronto para reconhecimento",
      detail: `${total}/40 imagens processadas`,
      total,
      icon: CheckCircle2,
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    };
  }

  if (complete) {
    return {
      label: "Imagens capturadas",
      detail: "Aguardando gerar embeddings",
      total,
      icon: ImageIcon,
      className: "border-blue-200 bg-blue-50 text-blue-700",
    };
  }

  return {
    label: "Captura pendente",
    detail: `${total}/40 imagens capturadas`,
    total,
    icon: Clock3,
    className: "border-amber-200 bg-amber-50 text-amber-700",
  };
}

export function StudentTable({ isLoading, onDelete, onSelect, selectedStudentId, statuses, students }: StudentTableProps) {
  return (
    <section className="overflow-hidden rounded-lg border border-border bg-card shadow-sm">
      <div className="border-b border-border px-4 py-4">
        <h3 className="text-base font-semibold">Lista de alunos</h3>
        <p className="text-sm text-muted-foreground">Dados academicos e andamento da biometria facial em um unico lugar.</p>
      </div>
      <div className="grid grid-cols-[1.05fr_0.65fr_0.75fr_1.1fr_120px] border-b border-border bg-muted/40 px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">
        <span>Nome</span>
        <span>Matricula</span>
        <span>Turma</span>
        <span>Status facial</span>
        <span>Acoes</span>
      </div>
      {isLoading ? <p className="p-4 text-sm text-muted-foreground">Carregando alunos...</p> : null}
      {!isLoading && students.length === 0 ? <p className="p-4 text-sm text-muted-foreground">Nenhum aluno encontrado.</p> : null}
      {students.map((student) => {
        const isSelected = student.id === selectedStudentId;
        const status = statuses?.[student.id];
        const summary = getBiometrySummary(status);
        const StatusIcon = summary.icon;

        return (
          <div
            key={student.id}
            className={`grid grid-cols-[1.05fr_0.65fr_0.75fr_1.1fr_120px] items-center gap-3 border-b border-border px-4 py-3 text-sm transition last:border-b-0 ${
              isSelected ? "bg-sky-50/70" : "hover:bg-muted/35"
            }`}
          >
            <div className="min-w-0">
              <p className="truncate font-semibold text-slate-950">{student.name}</p>
            </div>
            <span className="text-muted-foreground">{student.matricula ?? "-"}</span>
            <span className="text-muted-foreground">
              {[student.class_name, student.class_identifier].filter(Boolean).join(" ") || "-"}
            </span>
            <div className="min-w-0 space-y-2">
              <span
                className={`inline-flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold ${summary.className}`}
              >
                <StatusIcon className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{summary.label}</span>
              </span>
            </div>
            <div className="flex gap-1">
              <Button className="h-8 px-2" title="Abrir captura facial" variant="secondary" onClick={() => onSelect(student)}>
                <ScanFace className="h-4 w-4" />
              </Button>
              <button
                className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={() => onDelete(student.id)}
                title="Desativar aluno"
                type="button"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        );
      })}
    </section>
  );
}
