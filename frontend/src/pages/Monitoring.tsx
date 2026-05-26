import { CircleStop, Play, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { RealtimeEventsList } from "@/components/monitoring/RealtimeEventsList";
import { MonitoringStatusPanel } from "@/components/monitoring/MonitoringStatusPanel";
import { MonitoringVideoPanel } from "@/components/monitoring/MonitoringVideoPanel";
import { Button } from "@/components/ui/Button";
import { useMonitoring } from "@/hooks/useMonitoring";
import { useMonitoringOptions } from "@/hooks/useMonitoringOptions";

export function Monitoring() {
  const { data: options, isLoading: isLoadingOptions } = useMonitoringOptions();
  const { error, events, isActive, isLoading, refreshStatus, start, status, stop, websocketStatus } = useMonitoring();
  const [subjectId, setSubjectId] = useState("");
  const [classId, setClassId] = useState("");
  const [lessonType, setLessonType] = useState("Exposição");

  useEffect(() => {
    if (!options) {
      return;
    }
    setSubjectId((current) => current || String(options.subjects[0]?.id ?? ""));
    setClassId((current) => current || String(options.classes[0]?.id ?? ""));
    setLessonType((current) => current || options.lesson_types[0] || "Exposição");
  }, [options]);

  const selectedSubject = options?.subjects.find((subject) => String(subject.id) === subjectId);
  const selectedClass = options?.classes.find((classItem) => String(classItem.id) === classId);
  const canStart = Boolean(subjectId && classId && lessonType && !isActive && !isLoading);
  const canStop = Boolean(isActive && !isLoading);

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 rounded-lg border border-border bg-card p-5 shadow-sm xl:flex-row xl:items-center xl:justify-between">
        <div>
          <p className="text-sm font-medium text-primary">Tempo real</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight">Sala monitorada</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {selectedSubject?.nome ?? "Disciplina"} ·{" "}
            {[selectedClass?.nome, selectedClass?.identificador].filter(Boolean).join(" ") || "Turma"} · {lessonType}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => void refreshStatus()} disabled={isLoading}>
            <RefreshCw className="h-4 w-4" />
            Atualizar
          </Button>
          <Button
            disabled={!canStart}
            onClick={() => {
              void start({
                camera_id: 1,
                disciplina_id: Number(subjectId),
                turma_id: Number(classId),
                tipo_aula: lessonType,
              });
            }}
          >
            <Play className="h-4 w-4" />
            Iniciar
          </Button>
          <Button disabled={!canStop} onClick={() => void stop()} variant="secondary">
            <CircleStop className="h-4 w-4" />
            Parar
          </Button>
        </div>
      </header>

      <section className="grid gap-4 rounded-lg border border-border bg-card p-4 shadow-sm lg:grid-cols-[1fr_1fr_1fr]">
        <label className="space-y-2 text-sm font-medium">
          Disciplina
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
            disabled={isLoadingOptions || isActive}
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

        <label className="space-y-2 text-sm font-medium">
          Turma
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
            disabled={isLoadingOptions || isActive}
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

        <label className="space-y-2 text-sm font-medium">
          Tipo de aula
          <select
            className="h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
            disabled={isLoadingOptions || isActive}
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
      </section>

      {error || status.error ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error ?? status.error}</p>
      ) : null}

      <MonitoringStatusPanel isActive={isActive} status={status} />

      <section className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <MonitoringVideoPanel cameraStatus={status.camera_status} isActive={isActive} websocketStatus={websocketStatus} />
        <RealtimeEventsList events={events} />
      </section>
    </div>
  );
}
