import { Camera, CircleStop, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { MonitoringPanel } from "@/components/monitoring/MonitoringPanel";
import { Button } from "@/components/ui/Button";
import { useMonitoringOptions } from "@/hooks/useMonitoringOptions";
import { useAuthStore } from "@/stores/authStore";
import { useMonitoringStore } from "@/stores/monitoringStore";

export function Monitoring() {
  const token = useAuthStore((state) => state.token);
  const { data: options, isLoading: isLoadingOptions } = useMonitoringOptions();
  const activeSession = useMonitoringStore((state) => state.activeSession);
  const isLoadingSession = useMonitoringStore((state) => state.isLoading);
  const sessionError = useMonitoringStore((state) => state.error);
  const startSession = useMonitoringStore((state) => state.startSession);
  const stopSession = useMonitoringStore((state) => state.stopSession);
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

  const canStart = useMemo(
    () => Boolean(token && subjectId && classId && lessonType && !activeSession),
    [activeSession, classId, lessonType, subjectId, token],
  );

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Monitoramento</h1>
          <p className="text-sm text-muted-foreground">Base preparada para RTSP, IA no backend e WebSocket.</p>
        </div>
        <div className="flex gap-2">
          <Button
            disabled={!canStart || isLoadingSession}
            onClick={() => {
              if (!token || !subjectId || !classId) {
                return;
              }
              void startSession(token, {
                subject_id: Number(subjectId),
                class_id: Number(classId),
                lesson_type: lessonType,
              });
            }}
          >
            <Play className="h-4 w-4" />
            Iniciar
          </Button>
          <Button
            disabled={!token || !activeSession || isLoadingSession}
            onClick={() => {
              if (token) {
                void stopSession(token);
              }
            }}
            variant="secondary"
          >
            <CircleStop className="h-4 w-4" />
            Encerrar
          </Button>
        </div>
      </header>

      <section className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <div className="flex aspect-video min-h-[320px] items-center justify-center rounded-lg border border-border bg-slate-950 text-white">
          <div className="text-center">
            <Camera className="mx-auto h-10 w-10 opacity-70" />
            <p className="mt-3 text-sm opacity-80">Stream da câmera será entregue pelo backend</p>
          </div>
        </div>
        <MonitoringPanel
          activeSession={activeSession}
          classId={classId}
          error={sessionError}
          isLoadingOptions={isLoadingOptions}
          lessonType={lessonType}
          options={options}
          setClassId={setClassId}
          setLessonType={setLessonType}
          setSubjectId={setSubjectId}
          subjectId={subjectId}
        />
      </section>
    </div>
  );
}
