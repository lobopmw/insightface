import { ScanFace, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { Student } from "@/types/student";

type StudentTableProps = {
  isLoading?: boolean;
  students: Student[];
  onDelete: (studentId: string) => void;
  onSelect: (student: Student) => void;
};

export function StudentTable({ isLoading, onDelete, onSelect, students }: StudentTableProps) {
  return (
    <section className="overflow-hidden rounded-lg border border-border bg-card shadow-sm">
      <div className="border-b border-border px-4 py-4">
        <h3 className="text-base font-semibold">Lista de alunos</h3>
        <p className="text-sm text-muted-foreground">Selecione um aluno para cadastrar ou revisar a biometria facial.</p>
      </div>
      <div className="grid grid-cols-[1.2fr_0.8fr_0.9fr_120px] border-b border-border bg-muted/40 px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">
        <span>Nome</span>
        <span>Matricula</span>
        <span>Turma</span>
        <span>Acoes</span>
      </div>
      {isLoading ? <p className="p-4 text-sm text-muted-foreground">Carregando alunos...</p> : null}
      {!isLoading && students.length === 0 ? <p className="p-4 text-sm text-muted-foreground">Nenhum aluno encontrado.</p> : null}
      {students.map((student) => (
        <div
          key={student.id}
          className="grid grid-cols-[1.2fr_0.8fr_0.9fr_120px] items-center gap-3 border-b border-border px-4 py-3 text-sm transition hover:bg-muted/35 last:border-b-0"
        >
          <span className="font-medium">{student.name}</span>
          <span className="text-muted-foreground">{student.matricula ?? "-"}</span>
          <span className="text-muted-foreground">
            {[student.class_name, student.class_identifier].filter(Boolean).join(" ") || "-"}
          </span>
          <div className="flex gap-1">
            <Button className="h-8 px-2" variant="secondary" onClick={() => onSelect(student)}>
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
      ))}
    </section>
  );
}
