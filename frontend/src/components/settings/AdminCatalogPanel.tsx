import { Edit3, Link2, Plus, Save, Trash2, X } from "lucide-react";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/Button";
import {
  createAdminAssignment,
  createAdminClass,
  createAdminSubject,
  deleteAdminAssignment,
  deleteAdminClass,
  deleteAdminSubject,
  updateAdminClass,
  updateAdminStudent,
  updateAdminSubject,
} from "@/features/admin/catalogApi";
import { useAdminCatalog, useAdminStudents } from "@/hooks/useAdminCatalog";
import { useAuthStore } from "@/stores/authStore";
import type { AdminClass, AdminSubject } from "@/types/adminCatalog";
import type { Student } from "@/types/student";

type Tab = "subjects" | "classes" | "assignments" | "students";

export type AdminCatalogTab = Tab;

type AdminCatalogPanelProps = {
  activeTab?: Tab;
};

export function AdminCatalogPanel({ activeTab }: AdminCatalogPanelProps) {
  const token = useAuthStore((state) => state.token);
  const queryClient = useQueryClient();
  const { data: catalog } = useAdminCatalog();
  const { data: studentsData } = useAdminStudents();
  const [internalTab, setInternalTab] = useState<Tab>("subjects");
  const [subjectName, setSubjectName] = useState("");
  const [className, setClassName] = useState("");
  const [classIdentifier, setClassIdentifier] = useState("");
  const [assignment, setAssignment] = useState({ teacher_id: "", subject_id: "", class_id: "" });
  const [editingSubject, setEditingSubject] = useState<AdminSubject | null>(null);
  const [editingClass, setEditingClass] = useState<AdminClass | null>(null);
  const [editingStudent, setEditingStudent] = useState<Student | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["admin-catalog"] });
    void queryClient.invalidateQueries({ queryKey: ["admin-students"] });
    void queryClient.invalidateQueries({ queryKey: ["students"] });
    void queryClient.invalidateQueries({ queryKey: ["students", "classes"] });
  };

  const mutation = useMutation({
    mutationFn: async (action: () => Promise<unknown>) => action(),
    onSuccess: () => {
      setMessage("Alteração salva.");
      refresh();
    },
    onError: () => setMessage("Não foi possível salvar. Verifique se o item já existe ou está em uso."),
  });

  const tabs: Array<{ id: Tab; label: string }> = [
    { id: "subjects", label: "Disciplinas" },
    { id: "classes", label: "Classes" },
    { id: "assignments", label: "Vínculos" },
    { id: "students", label: "Alunos" },
  ];
  const tab = activeTab ?? internalTab;

  return (
    <section className="rounded-lg border border-border bg-card shadow-sm shadow-slate-200/60">
      <div className="border-b border-border px-4 py-4">
        <p className="text-xs font-semibold uppercase text-muted-foreground">Cadastros mestres</p>
        <h2 className="mt-1 text-base font-semibold">Elementos do sistema</h2>
        <p className="text-sm text-muted-foreground">Gerencie disciplinas, classes, vínculos de professor e ajustes de alunos.</p>
      </div>

      {!activeTab ? (
        <div className="flex gap-2 overflow-x-auto border-b border-border p-3">
          {tabs.map((item) => (
            <button
              key={item.id}
              className={`h-9 rounded-md px-3 text-sm font-semibold transition ${
                tab === item.id ? "bg-slate-950 text-white" : "bg-background text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
              onClick={() => setInternalTab(item.id)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}

      {message ? <p className="border-b border-border px-4 py-3 text-sm text-muted-foreground">{message}</p> : null}

      {tab === "subjects" ? (
        <div className="p-4">
          <form
            className="mb-4 flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              if (subjectName.trim()) {
                mutation.mutate(() => createAdminSubject(token as string, subjectName));
                setSubjectName("");
              }
            }}
          >
            <input className="h-10 flex-1 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setSubjectName(event.target.value)} placeholder="Nova disciplina" value={subjectName} />
            <Button disabled={!subjectName.trim() || mutation.isPending} type="submit"><Plus className="h-4 w-4" />Cadastrar</Button>
          </form>
          <div className="space-y-2">
            {(catalog?.subjects ?? []).map((subject) => (
              <div key={subject.id} className="flex flex-col gap-2 rounded-md border border-border p-3 sm:flex-row sm:items-center">
                {editingSubject?.id === subject.id ? (
                  <input className="h-9 flex-1 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setEditingSubject({ ...editingSubject, nome: event.target.value })} value={editingSubject.nome} />
                ) : (
                  <span className="flex-1 font-medium">{subject.nome}</span>
                )}
                <RowActions
                  isEditing={editingSubject?.id === subject.id}
                  onCancel={() => setEditingSubject(null)}
                  onDelete={() => mutation.mutate(() => deleteAdminSubject(token as string, subject.id))}
                  onEdit={() => setEditingSubject(subject)}
                  onSave={() => {
                    if (editingSubject) mutation.mutate(() => updateAdminSubject(token as string, subject.id, editingSubject.nome));
                    setEditingSubject(null);
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "classes" ? (
        <div className="p-4">
          <form
            className="mb-4 grid gap-2 sm:grid-cols-[1fr_0.8fr_auto]"
            onSubmit={(event) => {
              event.preventDefault();
              if (className.trim()) {
                mutation.mutate(() => createAdminClass(token as string, { nome: className, identificador: classIdentifier || null }));
                setClassName("");
                setClassIdentifier("");
              }
            }}
          >
            <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setClassName(event.target.value)} placeholder="Nome da classe" value={className} />
            <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setClassIdentifier(event.target.value)} placeholder="Identificador" value={classIdentifier} />
            <Button disabled={!className.trim() || mutation.isPending} type="submit"><Plus className="h-4 w-4" />Cadastrar</Button>
          </form>
          <div className="space-y-2">
            {(catalog?.classes ?? []).map((classItem) => (
              <div key={classItem.id} className="grid gap-2 rounded-md border border-border p-3 sm:grid-cols-[1fr_0.8fr_auto] sm:items-center">
                {editingClass?.id === classItem.id ? (
                  <>
                    <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setEditingClass({ ...editingClass, nome: event.target.value })} value={editingClass.nome} />
                    <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => setEditingClass({ ...editingClass, identificador: event.target.value })} value={editingClass.identificador ?? ""} />
                  </>
                ) : (
                  <>
                    <span className="font-medium">{classItem.nome}</span>
                    <span className="text-muted-foreground">{classItem.identificador || "-"}</span>
                  </>
                )}
                <RowActions
                  isEditing={editingClass?.id === classItem.id}
                  onCancel={() => setEditingClass(null)}
                  onDelete={() => mutation.mutate(() => deleteAdminClass(token as string, classItem.id))}
                  onEdit={() => setEditingClass(classItem)}
                  onSave={() => {
                    if (editingClass) mutation.mutate(() => updateAdminClass(token as string, classItem.id, { nome: editingClass.nome, identificador: editingClass.identificador || null }));
                    setEditingClass(null);
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "assignments" ? (
        <div className="p-4">
          <form
            className="mb-4 grid gap-2 lg:grid-cols-[1fr_1fr_1fr_auto]"
            onSubmit={(event) => {
              event.preventDefault();
              if (assignment.teacher_id && assignment.subject_id && assignment.class_id) {
                mutation.mutate(() => createAdminAssignment(token as string, {
                  teacher_id: Number(assignment.teacher_id),
                  subject_id: Number(assignment.subject_id),
                  class_id: Number(assignment.class_id),
                }));
              }
            }}
          >
            <Select value={assignment.teacher_id} onChange={(value) => setAssignment((current) => ({ ...current, teacher_id: value }))} placeholder="Professor" options={(catalog?.teachers ?? []).map((teacher) => ({ value: String(teacher.id), label: teacher.nome }))} />
            <Select value={assignment.subject_id} onChange={(value) => setAssignment((current) => ({ ...current, subject_id: value }))} placeholder="Disciplina" options={(catalog?.subjects ?? []).map((subject) => ({ value: String(subject.id), label: subject.nome }))} />
            <Select value={assignment.class_id} onChange={(value) => setAssignment((current) => ({ ...current, class_id: value }))} placeholder="Classe" options={(catalog?.classes ?? []).map((item) => ({ value: String(item.id), label: [item.nome, item.identificador].filter(Boolean).join(" - ") }))} />
            <Button disabled={!assignment.teacher_id || !assignment.subject_id || !assignment.class_id || mutation.isPending} type="submit"><Link2 className="h-4 w-4" />Vincular</Button>
          </form>
          <div className="space-y-2">
            {(catalog?.assignments ?? []).map((item) => (
              <div key={item.id} className="grid gap-2 rounded-md border border-border p-3 text-sm lg:grid-cols-[1fr_1fr_1fr_auto] lg:items-center">
                <span className="font-medium">{item.teacher_name}</span>
                <span className="text-muted-foreground">{item.subject_name}</span>
                <span className="text-muted-foreground">{[item.class_name, item.class_identifier].filter(Boolean).join(" - ")}</span>
                <button className="flex h-9 w-9 items-center justify-center rounded-md border border-rose-200 text-rose-700 hover:bg-rose-50 lg:justify-self-end" onClick={() => mutation.mutate(() => deleteAdminAssignment(token as string, item.id))} type="button">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "students" ? (
        <div className="space-y-2 p-4">
          {(studentsData?.items ?? []).map((student) => (
            <div key={student.id} className="grid gap-2 rounded-md border border-border p-3 text-sm lg:grid-cols-[1fr_0.7fr_0.8fr_auto] lg:items-center">
              {editingStudent?.id === student.id ? (
                <>
                  <input className="h-9 rounded-md border border-border bg-background px-3" onChange={(event) => setEditingStudent({ ...editingStudent, name: event.target.value })} value={editingStudent.name} />
                  <input className="h-9 rounded-md border border-border bg-background px-3" onChange={(event) => setEditingStudent({ ...editingStudent, matricula: event.target.value })} value={editingStudent.matricula ?? ""} />
                  <Select value={String(editingStudent.class_id ?? "")} onChange={(value) => setEditingStudent({ ...editingStudent, class_id: value ? Number(value) : null })} placeholder="Sem turma" options={(catalog?.classes ?? []).map((item) => ({ value: String(item.id), label: [item.nome, item.identificador].filter(Boolean).join(" - ") }))} />
                </>
              ) : (
                <>
                  <span className="font-medium">{student.name}</span>
                  <span className="text-muted-foreground">{student.matricula ?? "-"}</span>
                  <span className="text-muted-foreground">{[student.class_name, student.class_identifier].filter(Boolean).join(" - ") || "-"}</span>
                </>
              )}
              <RowActions
                isEditing={editingStudent?.id === student.id}
                onCancel={() => setEditingStudent(null)}
                onDelete={() => mutation.mutate(() => updateAdminStudent(token as string, student.id, { ativo: false }))}
                onEdit={() => setEditingStudent(student)}
                onSave={() => {
                  if (editingStudent) {
                    mutation.mutate(() => updateAdminStudent(token as string, student.id, {
                      name: editingStudent.name,
                      matricula: editingStudent.matricula,
                      class_id: editingStudent.class_id,
                    }));
                  }
                  setEditingStudent(null);
                }}
              />
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function Select({ onChange, options, placeholder, value }: { value: string; placeholder: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return (
    <select className="h-10 rounded-md border border-border bg-background px-3 text-sm" onChange={(event) => onChange(event.target.value)} value={value}>
      <option value="">{placeholder}</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>{option.label}</option>
      ))}
    </select>
  );
}

function RowActions({ isEditing, onCancel, onDelete, onEdit, onSave }: { isEditing: boolean; onCancel: () => void; onDelete: () => void; onEdit: () => void; onSave: () => void }) {
  return (
    <div className="flex gap-2 sm:justify-end">
      {isEditing ? (
        <>
          <button className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted" onClick={onCancel} type="button"><X className="h-4 w-4" /></button>
          <button className="flex h-9 w-9 items-center justify-center rounded-md bg-slate-950 text-white hover:bg-slate-800" onClick={onSave} type="button"><Save className="h-4 w-4" /></button>
        </>
      ) : (
        <>
          <button className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground hover:bg-muted" onClick={onEdit} type="button"><Edit3 className="h-4 w-4" /></button>
          <button className="flex h-9 w-9 items-center justify-center rounded-md border border-rose-200 text-rose-700 hover:bg-rose-50" onClick={onDelete} type="button"><Trash2 className="h-4 w-4" /></button>
        </>
      )}
    </div>
  );
}
