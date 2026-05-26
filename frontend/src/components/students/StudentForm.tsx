import { Save } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";
import type { StudentClass, StudentCreatePayload } from "@/types/student";

type StudentFormProps = {
  classes: StudentClass[];
  defaultRegistration?: string;
  isLoading?: boolean;
  onSubmit: (payload: StudentCreatePayload) => void;
};

export function StudentForm({ classes, defaultRegistration, isLoading, onSubmit }: StudentFormProps) {
  const [name, setName] = useState("");
  const [matricula, setMatricula] = useState("");
  const [classId, setClassId] = useState("");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    setMatricula((current) => current || defaultRegistration || "");
    setClassId((current) => current || String(classes[0]?.id ?? ""));
  }, [classes, defaultRegistration]);

  return (
    <form
      className="rounded-lg border border-border bg-card p-5 shadow-sm"
      onSubmit={(event) => {
        event.preventDefault();
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
        <p className="text-sm text-muted-foreground">Preencha os dados acadêmicos antes da captura facial.</p>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <label className="space-y-2 text-sm font-medium">
          Nome
          <input
            className="h-10 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
            onChange={(event) => setName(event.target.value)}
            value={name}
          />
        </label>
        <label className="space-y-2 text-sm font-medium">
          Matricula
          <input
            className="h-10 w-full rounded-md border border-border bg-background px-3 outline-none focus:ring-2 focus:ring-primary/30"
            onChange={(event) => setMatricula(event.target.value)}
            value={matricula}
          />
        </label>
        <label className="space-y-2 text-sm font-medium">
          Turma / classe
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3"
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
        </label>
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

      <div className="mt-5 flex justify-end">
        <Button disabled={!name.trim() || isLoading} type="submit">
          <Save className="h-4 w-4" />
          Salvar aluno
        </Button>
      </div>
    </form>
  );
}
