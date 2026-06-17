import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import settings

try:
    import cv2
except Exception:  # pragma: no cover - depends on runtime optional package
    cv2 = None


@dataclass
class RTSPCaptureStatus:
    status: str
    source: str
    connected: bool
    checked_at: datetime
    frames_captured: int = 0
    error: str | None = None


class RTSPCaptureService:
    """Basic OpenCV/FFmpeg capture service used until the AI runtime is extracted."""

    def __init__(self, relay_host: str | None = None, relay_port: int | None = None, rtsp_url: str | None = None):
        self.relay_host = relay_host or settings.RELAY_HOST
        self.relay_port = relay_port or settings.RELAY_PORT
        self.rtsp_url = rtsp_url or settings.RTSP_URL
        self._capture = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._last_frame_jpeg: bytes | None = None
        self._frames_captured = 0
        self._status = RTSPCaptureStatus(
            status="idle",
            source=self.source_label,
            connected=False,
            checked_at=datetime.now(UTC),
        )

    @property
    def source_label(self) -> str:
        if self.rtsp_url:
            return self.rtsp_url
        return f"tcp://{self.relay_host}:{self.relay_port}"

    def start(self) -> RTSPCaptureStatus:
        if cv2 is None:
            self._status = RTSPCaptureStatus(
                status="error",
                source=self.source_label,
                connected=False,
                checked_at=datetime.now(UTC),
                error="opencv_not_installed",
            )
            return self._status

        if not self.rtsp_url:
            self._status = RTSPCaptureStatus(
                status="error",
                source=self.source_label,
                connected=False,
                checked_at=datetime.now(UTC),
                error="RTSP_URL ausente",
            )
            return self._status

        if self._running:
            return self.get_status()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            status = self.get_status()
            if status.connected or status.error:
                return status
            time.sleep(0.05)
        return self.get_status()

    def _open_capture(self):
        if cv2 is None:
            return None
        capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not capture.isOpened():
            try:
                capture.release()
            except Exception:
                pass
            return None
        return capture

    def _capture_loop(self) -> None:
        reconnect_delay = 1.5
        while self._running:
            self._capture = self._open_capture()
            if self._capture is None:
                self._set_status("unavailable", False, "OpenCV não conseguiu abrir o stream")
                time.sleep(reconnect_delay)
                continue

            self._set_status("active", True, None)
            while self._running and self._capture is not None:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._set_status("reconnecting", False, "Falha ao ler frame")
                    break

                encoded, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if encoded:
                    with self._lock:
                        self._last_frame_jpeg = jpg.tobytes()
                        self._frames_captured += 1
                        self._status = RTSPCaptureStatus(
                            status="active",
                            source=self.source_label,
                            connected=True,
                            checked_at=datetime.now(UTC),
                            frames_captured=self._frames_captured,
                        )

            self._release_capture()
            if self._running:
                time.sleep(reconnect_delay)

    def _set_status(self, status: str, connected: bool, error: str | None) -> None:
        with self._lock:
            self._status = RTSPCaptureStatus(
                status=status,
                source=self.source_label,
                connected=connected,
                checked_at=datetime.now(UTC),
                frames_captured=self._frames_captured,
                error=error,
            )

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def stop(self) -> RTSPCaptureStatus:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release_capture()
        with self._lock:
            self._last_frame_jpeg = None
            self._status = RTSPCaptureStatus(
                status="idle",
                source=self.source_label,
                connected=False,
                checked_at=datetime.now(UTC),
                frames_captured=self._frames_captured,
            )
            return self._status

    def get_status(self) -> RTSPCaptureStatus:
        with self._lock:
            return self._status

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._last_frame_jpeg

    def mjpeg_frames(self):
        while self._running:
            frame = self.get_latest_jpeg()
            if frame is None:
                time.sleep(0.1)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
