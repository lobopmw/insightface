import { Plus, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/Button";
import { createStudent, deactivateStudent } from "@/features/students/studentsApi";
import { useNextRegistration, useStudentClasses, useStudents } from "@/hooks/useStudents";
import { useAuthStore } from "@/stores/authStore";

export function Students() {
  const token = useAuthStore((state) => state.token);
  const queryClient = useQueryClient();
  const { data, isLoading, isError } = useStudents();
  const { data: classesData } = useStudentClasses();
  const { data: nextRegistration, refetch: refetchNextRegistration } = useNextRegistration();
  const students = data?.items ?? [];
  const classes = classesData?.items ?? [];
  const [name, setName] = useState("");
  const [matricula, setMatricula] = useState("");
  const [classId, setClassId] = useState("");

  useEffect(() => {
    if (!matricula && nextRegistration?.matricula) {
      setMatricula(nextRegistration.matricula);
    }
    if (!classId && classes[0]?.id) {
      setClassId(String(classes[0].id));
    }
  }, [classId, classes, matricula, nextRegistration]);

  const createMutation = useMutation({
    mutationFn: () =>
      createStudent(token as string, {
        name,
        matricula,
        class_id: classId ? Number(classId) : null,
      }),
    onSuccess: () => {
      setName("");
      setMatricula("");
      void queryClient.invalidateQueries({ queryKey: ["students"] });
      void refetchNextRegistration();
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (studentId: string) => deactivateStudent(token as string, studentId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["students"] });
    },
  });

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Alunos</h1>
          <p className="text-sm text-muted-foreground">Lista inicial consultando o banco atual.</p>
        </div>
        <Button variant="secondary" onClick={() => void refetchNextRegistration()}>
          <RefreshCw className="h-4 w-4" />
          Matrícula
        </Button>
      </header>

      <form
        className="grid gap-3 rounded-lg border border-border bg-card p-4 lg:grid-cols-[1.2fr_0.8fr_0.9fr_auto]"
        onSubmit={(event) => {
          event.preventDefault();
          if (token && name.trim()) {
            createMutation.mutate();
          }
        }}
      >
        <input
          className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          onChange={(event) => setName(event.target.value)}
          placeholder="Nome do aluno"
          value={name}
        />
        <input
          className="h-10 rounded-md border border-border bg-background px-3 text-sm outline-none focus:ring-2 focus:ring-primary/30"
          onChange={(event) => setMatricula(event.target.value)}
          placeholder="Matrícula"
          value={matricula}
        />
        <select
          className="h-10 rounded-md border border-border bg-background px-3 text-sm"
          onChange={(event) => setClassId(event.target.value)}
          value={classId}
        >
          <option value="">Sem turma</option>
          {classes.map((classItem) => (
            <option key={classItem.id} value={classItem.id}>
              {[classItem.nome, classItem.identificador].filter(Boolean).join(" ")}
            </option>
          ))}
        </select>
        <Button disabled={!token || !name.trim() || createMutation.isPending} type="submit">
          <Plus className="h-4 w-4" />
          Adicionar
        </Button>
        {createMutation.isError ? (
          <p className="text-sm text-red-700 lg:col-span-4">Não foi possível cadastrar o aluno.</p>
        ) : null}
      </form>

      <section className="overflow-hidden rounded-lg border border-border bg-card">
        <div className="grid grid-cols-[1.4fr_1fr_1fr_44px] border-b border-border px-4 py-3 text-xs font-semibold uppercase text-muted-foreground">
          <span>Nome</span>
          <span>Matrícula</span>
          <span>Turma</span>
          <span />
        </div>

        {isLoading ? <p className="p-4 text-sm text-muted-foreground">Carregando alunos...</p> : null}
        {isError ? <p className="p-4 text-sm text-red-700">Não foi possível carregar alunos.</p> : null}
        {!isLoading && !isError && students.length === 0 ? (
          <p className="p-4 text-sm text-muted-foreground">Nenhum aluno encontrado para o escopo atual.</p>
        ) : null}

        {students.map((student) => (
          <div
            key={student.id}
            className="grid grid-cols-[1.4fr_1fr_1fr_44px] gap-3 border-b border-border px-4 py-3 text-sm last:border-b-0"
          >
            <span className="font-medium">{student.name}</span>
            <span className="text-muted-foreground">{student.matricula ?? "-"}</span>
            <span className="text-muted-foreground">
              {[student.class_name, student.class_identifier].filter(Boolean).join(" ") || "-"}
            </span>
            <button
              className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
              disabled={deactivateMutation.isPending}
              onClick={() => deactivateMutation.mutate(student.id)}
              title="Desativar aluno"
              type="button"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </section>
    </div>
  );
}
