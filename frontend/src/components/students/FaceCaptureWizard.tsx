import { ArrowRight, CheckCircle2, RotateCcw } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import {
  CameraCapturePanel,
  type CameraCapturePanelHandle,
} from "@/components/students/CameraCapturePanel";
import { PoseProgressCard, POSE_LABELS, POSE_ORDER } from "@/components/students/PoseProgressCard";
import type { PoseKey, Student, StudentFaceStatus } from "@/types/student";

const CAPTURE_TARGET = 10;
const CAPTURE_INTERVAL_SECONDS = 3;

const POSE_INSTRUCTIONS: Record<PoseKey, string> = {
  frontal: "Posicione o rosto de frente para a câmera.",
  lateral_esquerda: "Vire levemente para a esquerda.",
  lateral_direita: "Vire levemente para a direita.",
  cabeca_baixa: "Incline a cabeça levemente para baixo.",
};

function wait(seconds: number, onTick: (value: number) => void, shouldCancel: () => boolean) {
  return new Promise<void>((resolve) => {
    let remaining = seconds;
    onTick(remaining);
    const interval = window.setInterval(() => {
      if (shouldCancel()) {
        window.clearInterval(interval);
        resolve();
        return;
      }
      remaining -= 1;
      onTick(Math.max(remaining, 0));
      if (remaining <= 0) {
        window.clearInterval(interval);
        resolve();
      }
    }, 1000);
  });
}

export function FaceCaptureWizard({
  isUploading,
  onUploadImage,
  selectedStudent,
  status,
}: {
  isUploading?: boolean;
  onUploadImage: (pose: PoseKey, image: Blob, index: number) => Promise<void>;
  selectedStudent?: Student | null;
  status?: StudentFaceStatus;
}) {
  const cameraRef = useRef<CameraCapturePanelHandle | null>(null);
  const cancelRef = useRef(false);
  const [activePose, setActivePose] = useState<PoseKey>("frontal");
  const [isCapturing, setIsCapturing] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [captureIndex, setCaptureIndex] = useState(0);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const activeIndex = POSE_ORDER.indexOf(activePose);
  const storedCount = status?.[activePose]?.count ?? 0;
  const visualCount = Math.min(CAPTURE_TARGET, Math.max(storedCount, captureIndex));
  const poseComplete = Boolean(status?.[activePose]?.complete);
  const allComplete = useMemo(() => POSE_ORDER.every((pose) => status?.[pose]?.complete), [status]);
  const canGoNext = poseComplete && !isCapturing && activeIndex < POSE_ORDER.length - 1;

  const startAutomaticCapture = async () => {
    if (!selectedStudent || isCapturing) {
      return;
    }
    cancelRef.current = false;
    setCaptureError(null);
    setIsCapturing(true);
    setCaptureIndex(0);

    try {
      for (let nextIndex = 1; nextIndex <= CAPTURE_TARGET; nextIndex += 1) {
        await wait(CAPTURE_INTERVAL_SECONDS, setCountdown, () => cancelRef.current);
        if (cancelRef.current) {
          break;
        }

        const image = await cameraRef.current?.captureFrame();
        if (!image) {
          throw new Error("Nao foi possivel capturar o frame atual da camera.");
        }
        await onUploadImage(activePose, image, nextIndex);
        setCaptureIndex(nextIndex);
      }
    } catch (error) {
      setCaptureError(error instanceof Error ? error.message : "Falha durante a captura automatica.");
    } finally {
      setCountdown(null);
      setIsCapturing(false);
      cancelRef.current = false;
    }
  };

  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <p className="text-sm font-medium text-primary">Etapa 2</p>
        <h3 className="mt-1 text-base font-semibold">Captura facial automática</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          {selectedStudent ? `Aluno: ${selectedStudent.name}` : "Salve ou selecione um aluno para habilitar a captura."}
        </p>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <h4 className="text-2xl font-semibold">{POSE_LABELS[activePose]}</h4>
                <p className="mt-1 text-sm text-muted-foreground">{POSE_INSTRUCTIONS[activePose]}</p>
              </div>
              <span className="rounded-md bg-muted px-3 py-1 text-sm font-medium text-muted-foreground">
                {visualCount}/{CAPTURE_TARGET} imagens
              </span>
            </div>

            <div className="mt-4 h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${Math.round((visualCount / CAPTURE_TARGET) * 100)}%` }}
              />
            </div>

            <div className="mt-4 rounded-md border border-border bg-background px-3 py-2 text-sm">
              {isCapturing ? (
                <div className="space-y-1">
                  <p className="font-medium">Capturando imagem {Math.min(captureIndex + 1, CAPTURE_TARGET)} de {CAPTURE_TARGET}</p>
                  <p className="text-muted-foreground">Próxima captura em {countdown ?? CAPTURE_INTERVAL_SECONDS} segundos</p>
                </div>
              ) : poseComplete ? (
                <p className="flex items-center gap-2 font-medium text-emerald-700">
                  <CheckCircle2 className="h-4 w-4" />
                  Pose concluída
                </p>
              ) : (
                <p className="text-muted-foreground">Clique em Capturar para registrar 10 imagens automaticamente.</p>
              )}
            </div>
          </div>

          <CameraCapturePanel
            ref={cameraRef}
            activePose={activePose}
            captureLabel={poseComplete ? "Capturar novamente" : "Capturar"}
            disabled={!selectedStudent || isUploading}
            isCapturing={isCapturing}
            onCancel={() => {
              cancelRef.current = true;
            }}
            onCaptureClick={() => {
              void startAutomaticCapture();
            }}
          />

          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                setCaptureIndex(0);
                setCaptureError(null);
              }}
              disabled={isCapturing}
            >
              <RotateCcw className="h-4 w-4" />
              Resetar visual
            </Button>
            <Button
              variant="secondary"
              disabled={!canGoNext}
              onClick={() => {
                setCaptureIndex(0);
                setCaptureError(null);
                setActivePose(POSE_ORDER[Math.min(activeIndex + 1, POSE_ORDER.length - 1)]);
              }}
            >
              <ArrowRight className="h-4 w-4" />
              Próximo
            </Button>
          </div>

          {captureError ? (
            <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{captureError}</p>
          ) : null}

          {allComplete ? (
            <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
              Todas as poses foram concluídas. Aluno pronto para gerar embeddings.
            </p>
          ) : null}
        </div>

        <PoseProgressCard activePose={activePose} status={status} targetCount={CAPTURE_TARGET} />
      </div>
    </section>
  );
}
