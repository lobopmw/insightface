import { CircleStop, Play, RefreshCw, SlidersHorizontal, Video } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { MonitoringStatusPanel } from "@/components/monitoring/MonitoringStatusPanel";
import { MonitoringVideoPanel } from "@/components/monitoring/MonitoringVideoPanel";
import { Button } from "@/components/ui/Button";
import { useMonitoring } from "@/hooks/useMonitoring";
import { useMonitoringOptions } from "@/hooks/useMonitoringOptions";
import { formatMonitoringError } from "@/lib/monitoringErrors";

export function Monitoring() {
  const { data: options, isLoading: isLoadingOptions } = useMonitoringOptions();
  const { error, isActive, isLoading, refreshStatus, start, status, stop, websocketStatus } = useMonitoring();
  const [subjectId, setSubjectId] = useState("");
  const [classId, setClassId] = useState("");
  const [lessonType, setLessonType] = useState("Exposição");

  useEffect(() => {
    if (!options) {
      return;
    }
    setSubjectId((current) => current || String(options.subjects[0]?.id ?? ""));
    setLessonType((current) => current || options.lesson_types[0] || "Exposição");
  }, [options]);

  const availableClasses = useMemo(() => {
    if (!options) {
      return [];
    }
    const assignments = options.assignments ?? [];
    if (!subjectId || assignments.length === 0) {
      return options.classes;
    }
    const classIds = new Set(assignments.filter((item) => String(item.subject_id) === subjectId).map((item) => item.class_id));
    return options.classes.filter((classItem) => classIds.has(classItem.id));
  }, [options, subjectId]);

  useEffect(() => {
    if (availableClasses.length === 0) {
      setClassId("");
      return;
    }
    setClassId((current) => {
      if (current && availableClasses.some((classItem) => String(classItem.id) === current)) {
        return current;
      }
      return String(availableClasses[0].id);
    });
  }, [availableClasses]);

  const selectedSubject = options?.subjects.find((subject) => String(subject.id) === subjectId);
  const selectedClass = availableClasses.find((classItem) => String(classItem.id) === classId);
  const canStart = Boolean(subjectId && classId && lessonType && !isActive && !isLoading);
  const canStop = Boolean(isActive && !isLoading);
  const monitoringError = error ?? status.error;
  const monitoringErrorMessage = formatMonitoringError(monitoringError);

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 rounded-lg border border-border bg-card p-5 shadow-sm shadow-slate-200/60 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <p className="inline-flex items-center gap-2 text-xs font-semibold uppercase text-primary">
            <Video className="h-4 w-4" />
            Tempo real
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">Sala monitorada</h2>
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

      <section className="rounded-lg border border-border bg-card p-4 shadow-sm shadow-slate-200/60">
        <div className="mb-4 flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Parâmetros da sessão</h3>
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_1fr_1fr]">
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
              {availableClasses.length === 0 ? <option value="">Nenhuma turma vinculada</option> : null}
              {availableClasses.map((classItem) => (
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
        </div>
      </section>

      {monitoringErrorMessage ? (
        <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700" title={monitoringError ?? undefined}>
          {monitoringErrorMessage}
        </p>
      ) : null}

      <MonitoringStatusPanel isActive={isActive} status={status} />

      <section>
        <MonitoringVideoPanel cameraStatus={status.camera_status} isActive={isActive} websocketStatus={websocketStatus} />
      </section>
    </div>
  );
}
