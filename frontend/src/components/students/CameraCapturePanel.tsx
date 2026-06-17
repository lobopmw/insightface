import { Camera, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useImperativeHandle } from "react";
import { forwardRef } from "react";

import { Button } from "@/components/ui/Button";
import { useFaceCapture } from "@/hooks/useFaceCapture";
import type { PoseKey } from "@/types/student";
import { POSE_LABELS } from "@/components/students/PoseProgressCard";

type CameraCapturePanelProps = {
  activePose: PoseKey;
  captureLabel?: string;
  disabled?: boolean;
  isCapturing?: boolean;
  onCancel?: () => void;
  onCaptureClick: () => void;
};

export type CameraCapturePanelHandle = {
  captureFrame: () => Promise<Blob | null>;
};

export const CameraCapturePanel = forwardRef<CameraCapturePanelHandle, CameraCapturePanelProps>(function CameraCapturePanel(
  { activePose, captureLabel = "Capturar", disabled, isCapturing, onCancel, onCaptureClick },
  ref,
) {
  const { cameraError, captureFrame, isCameraReady, startCamera, stopCamera, videoRef } = useFaceCapture();

  useEffect(() => {
    void startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  useImperativeHandle(ref, () => ({ captureFrame }), [captureFrame]);

  return (
    <section className="rounded-lg border border-border bg-card p-3 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">Câmera</h3>
          <p className="text-xs text-muted-foreground">Pose: {POSE_LABELS[activePose]}</p>
        </div>
        <Button className="h-9 px-3" variant="secondary" disabled={isCapturing} onClick={() => void startCamera()}>
          <RefreshCw className="h-4 w-4" />
          Câmera
        </Button>
      </div>

      <div className="mx-auto mt-3 max-w-[760px] overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
        <video ref={videoRef} className="aspect-[16/7] max-h-[360px] w-full object-cover" muted playsInline />
      </div>

      {cameraError ? <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{cameraError}</p> : null}

      <div className="mt-3 flex flex-wrap gap-2">
        <Button className="h-9 px-3" disabled={disabled || !isCameraReady || isCapturing} onClick={onCaptureClick}>
          <Camera className="h-4 w-4" />
          {captureLabel}
        </Button>
        {isCapturing ? (
          <Button className="h-9 px-3" variant="secondary" onClick={onCancel}>
            <XCircle className="h-4 w-4" />
            Cancelar captura
          </Button>
        ) : null}
      </div>
    </section>
  );
});
