import { Camera, Wifi, WifiOff } from "lucide-react";

import { buildApiUrl } from "@/services/api";

type MonitoringVideoPanelProps = {
  cameraStatus: string;
  isActive: boolean;
  websocketStatus: string;
};

export function MonitoringVideoPanel({ cameraStatus, isActive, websocketStatus }: MonitoringVideoPanelProps) {
  const isWebSocketConnected = websocketStatus === "connected";

  return (
    <section className="relative aspect-video min-h-[320px] overflow-hidden rounded-lg border border-slate-800 bg-slate-950 text-white shadow-sm shadow-slate-300/60">
      <div className="absolute left-4 top-4 z-10 flex flex-wrap gap-2">
        <span className="inline-flex items-center gap-2 rounded-md bg-black/45 px-3 py-1.5 text-xs font-medium backdrop-blur">
          <Camera className="h-3.5 w-3.5" />
          {cameraStatus}
        </span>
        <span className="inline-flex items-center gap-2 rounded-md bg-black/45 px-3 py-1.5 text-xs font-medium backdrop-blur">
          {isWebSocketConnected ? (
            <Wifi className="h-3.5 w-3.5 text-emerald-300" />
          ) : (
            <WifiOff className="h-3.5 w-3.5 text-amber-300" />
          )}
          WebSocket {websocketStatus}
        </span>
      </div>

      {isActive ? (
        <img
          alt="Video do monitoramento"
          className="h-full w-full object-contain"
          src={`${buildApiUrl("/monitoring/video-feed")}?t=${Date.now()}`}
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-[linear-gradient(135deg,rgba(8,47,73,0.22),rgba(15,23,42,0.94))]">
          <div className="text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-lg border border-white/15 bg-white/10">
              <Camera className="h-8 w-8 opacity-80" />
            </div>
            <p className="mt-4 text-sm text-white/75">Área principal para vídeo da câmera</p>
            <p className="mt-1 text-xs text-white/50">Inicie uma sessão para abrir o MJPEG do backend</p>
          </div>
        </div>
      )}
    </section>
  );
}
