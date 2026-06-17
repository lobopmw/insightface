import { Save, UserCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import type { Student, StudentClass, StudentCreatePayload } from "@/types/student";

type StudentFormProps = {
  classes: StudentClass[];
  defaultRegistration?: string;
  isLoading?: boolean;
  onSelectExisting: (student: Student) => void;
  onSubmit: (payload: StudentCreatePayload) => void;
  selectedStudent?: Student | null;
  students: Student[];
};

export function StudentForm({
  classes,
  defaultRegistration,
  isLoading,
  onSelectExisting,
  onSubmit,
  selectedStudent,
  students,
}: StudentFormProps) {
  const [mode, setMode] = useState<"new" | "existing">("new");
  const [name, setName] = useState("");
  const [matricula, setMatricula] = useState("");
  const [classId, setClassId] = useState("");
  const [existingStudentId, setExistingStudentId] = useState("");
  const [notes, setNotes] = useState("");

  const studentsForClass = useMemo(() => {
    if (!classId) {
      return students;
    }
    return students.filter((student) => String(student.class_id ?? "") === classId);
  }, [classId, students]);

  useEffect(() => {
    setMatricula(defaultRegistration || "");
  }, [defaultRegistration]);

  useEffect(() => {
    setClassId((current) => current || String(classes[0]?.id ?? ""));
  }, [classes]);

  useEffect(() => {
    if (mode !== "existing") {
      return;
    }
    const nextStudent = studentsForClass.find((student) => student.id === existingStudentId) ?? studentsForClass[0];
    setExistingStudentId(nextStudent?.id ?? "");
  }, [existingStudentId, mode, studentsForClass]);

  const selectedExistingStudent = students.find((student) => student.id === existingStudentId);
  const hasExistingStudents = students.length > 0;

  return (
    <form
      className="rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60"
      onSubmit={(event) => {
        event.preventDefault();
        if (mode === "existing") {
          if (selectedExistingStudent) {
            onSelectExisting(selectedExistingStudent);
          }
          return;
        }
        onSubmit({
          name,
          matricula,
          class_id: classId ? Number(classId) : null,
        });
      }}
    >
      <div>
        <p className="text-sm font-medium text-primary">Etapa 1</p>
        <h3 className="mt-1 text-base font-semibold">Dados do aluno</h3>
        <p className="text-sm text-muted-foreground">Escolha entre cadastrar um aluno novo ou atualizar a captura de um aluno existente.</p>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-2 rounded-md border border-border bg-background p-1">
        <button
          className={`h-10 rounded-md text-sm font-semibold transition ${
            mode === "new" ? "bg-slate-950 text-white shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
          onClick={() => setMode("new")}
          type="button"
        >
          Aluno novo
        </button>
        <button
          className={`h-10 rounded-md text-sm font-semibold transition ${
            mode === "existing" ? "bg-slate-950 text-white shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground"
          }`}
          disabled={!hasExistingStudents}
          onClick={() => setMode("existing")}
          type="button"
        >
          Aluno já cadastrado
        </button>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <label className="space-y-2 text-sm font-medium">
          Turma / classe
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3"
            onChange={(event) => {
              setClassId(event.target.value);
              setExistingStudentId("");
            }}
            value={classId}
          >
            <option value="">Sem turma</option>
            {classes.map((classItem) => (
              <option key={classItem.id} value={classItem.id}>
                {[classItem.nome, classItem.identificador].filter(Boolean).join(" ")}
              </option>
            ))}
          </select>
        </label>

        {mode === "existing" ? (
          <label className="space-y-2 text-sm font-medium">
            Selecione o aluno
            <select
              className="h-10 w-full rounded-md border border-border bg-background px-3"
              disabled={studentsForClass.length === 0}
              onChange={(event) => setExistingStudentId(event.target.value)}
              value={existingStudentId}
            >
              {studentsForClass.length === 0 ? <option value="">Nenhum aluno nesta turma</option> : null}
              {studentsForClass.map((student) => (
                <option key={student.id} value={student.id}>
                  {[student.name, student.matricula].filter(Boolean).join(" - ")}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <>
            <label className="space-y-2 text-sm font-medium">
              Nome
              <input
                className="h-10 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
                onChange={(event) => setName(event.target.value)}
                placeholder="Digite o nome completo"
                value={name}
              />
            </label>
            <label className="space-y-2 text-sm font-medium">
              Matrícula
              <input
                className="h-10 w-full rounded-md border border-border bg-muted px-3 text-muted-foreground outline-none"
                placeholder="Gerada automaticamente"
                readOnly
                value={matricula}
              />
            </label>
          </>
        )}

        <label className="space-y-2 text-sm font-medium">
          Observacoes
          <input
            className="h-10 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Opcional, nao enviado nesta etapa"
            value={notes}
          />
        </label>
      </div>

      {mode === "existing" ? (
        <div className="mt-4 rounded-md border border-cyan-100 bg-cyan-50 px-3 py-2 text-sm text-cyan-800">
          {selectedExistingStudent
            ? `Aluno selecionado: ${selectedExistingStudent.name}. A captura facial será atualizada para este cadastro.`
            : selectedStudent
              ? `Aluno atual: ${selectedStudent.name}.`
              : "Selecione um aluno já cadastrado para atualizar as imagens faciais."}
        </div>
      ) : null}

      <div className="mt-5 flex justify-end">
        <Button disabled={mode === "new" ? !name.trim() || isLoading : !selectedExistingStudent} type="submit">
          {mode === "new" ? <Save className="h-4 w-4" /> : <UserCheck className="h-4 w-4" />}
          {mode === "new" ? "Salvar aluno" : "Usar aluno selecionado"}
        </Button>
      </div>
    </form>
  );
}
