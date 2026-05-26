import { useCallback, useRef, useState } from "react";

export function useFaceCapture() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    if (streamRef.current) {
      setIsCameraReady(true);
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "A webcam do navegador não está disponível. Use localhost ou HTTPS, ou verifique as permissões da câmera.",
      );
      setIsCameraReady(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 720, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (error) {
          const message = error instanceof Error ? error.message : "";
          if (!message.toLowerCase().includes("interrupted by a new load request")) {
            throw error;
          }
        }
      }
      setIsCameraReady(true);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Nao foi possivel acessar a camera");
      setIsCameraReady(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsCameraReady(false);
  }, []);

  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      return null;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.86);
    });
  }, []);

  return {
    cameraError,
    captureFrame,
    isCameraReady,
    startCamera,
    stopCamera,
    videoRef,
  };
}
