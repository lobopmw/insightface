import type { MonitoringOptions, MonitoringSession } from "@/types/monitoring";

type MonitoringPanelProps = {
  activeSession: MonitoringSession | null;
  classId: string;
  error: string | null;
  isLoadingOptions: boolean;
  lessonType: string;
  options?: MonitoringOptions;
  setClassId: (value: string) => void;
  setLessonType: (value: string) => void;
  setSubjectId: (value: string) => void;
  subjectId: string;
};

export function MonitoringPanel({
  activeSession,
  classId,
  error,
  isLoadingOptions,
  lessonType,
  options,
  setClassId,
  setLessonType,
  setSubjectId,
  subjectId,
}: MonitoringPanelProps) {
  return (
    <aside className="rounded-lg border border-border bg-card p-4">
      <h2 className="text-base font-semibold">Sessão</h2>
      <div className="mt-4 space-y-4 text-sm">
        <label className="space-y-2 font-medium">
          Disciplina
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3"
            disabled={isLoadingOptions || Boolean(activeSession)}
            onChange={(event) => setSubjectId(event.target.value)}
            value={subjectId}
          >
            {(options?.subjects ?? []).map((subject) => (
              <option key={subject.id} value={subject.id}>
                {subject.nome}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2 font-medium">
          Turma
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3"
            disabled={isLoadingOptions || Boolean(activeSession)}
            onChange={(event) => setClassId(event.target.value)}
            value={classId}
          >
            {(options?.classes ?? []).map((classItem) => (
              <option key={classItem.id} value={classItem.id}>
                {[classItem.nome, classItem.identificador].filter(Boolean).join(" ")}
              </option>
            ))}
          </select>
        </label>

        <label className="space-y-2 font-medium">
          Tipo de aula
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3"
            disabled={isLoadingOptions || Boolean(activeSession)}
            onChange={(event) => setLessonType(event.target.value)}
            value={lessonType}
          >
            {(options?.lesson_types ?? ["Exposição"]).map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </label>

        <dl className="space-y-3 border-t border-border pt-4">
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Status</dt>
            <dd className="font-medium">{activeSession ? activeSession.status : "Aguardando"}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Sessão</dt>
            <dd className="font-medium">{activeSession?.id ?? "-"}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Câmera</dt>
            <dd className="font-medium">RTSP pendente</dd>
          </div>
        </dl>

        {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p> : null}
      </div>
    </aside>
  );
}
