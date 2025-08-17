# video_stream.py
import cv2
import threading
import time

class VideoStream:
    """
    Leitor de vídeo em thread, sempre mantendo o frame mais novo.
    - Tenta GStreamer (pipeline low-latency) primeiro.
    - Se falhar, cai para FFMPEG com BUFFERSIZE=1 (no-buffer).
    A API é compatível: VideoStream(src).start() / .read() / .stop()
    """
    def __init__(self, src, width=None, height=None, gst_latency_ms=0, use_tcp=True):
        self.src = src
        self.width = width
        self.height = height
        self.gst_latency_ms = max(0, int(gst_latency_ms))
        self.use_tcp = use_tcp

        self.cap = None
        self.backend = None  # "gstreamer" | "ffmpeg" | "other"
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def _open_gstreamer(self):
        # protocols: 0x00000001 udp, 0x00000002 tcp (gstreamer usa flags; mais simples via string "tcp"/"udp")
        proto = "tcp" if self.use_tcp else "udp"
        pipeline = (
            f"rtspsrc location={self.src} latency={self.gst_latency_ms} protocols={proto} ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! appsink sync=false drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cap if cap.isOpened() else None

    def _open_ffmpeg(self):
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        # reduzir latência/filas
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _open(self):
        if isinstance(self.src, str) and self.src.lower().startswith("rtsp"):
            # 1) tentar GStreamer
            cap = self._open_gstreamer()
            if cap is not None:
                self.backend = "gstreamer"
                return cap
            # 2) fallback: FFMPEG
            cap = self._open_ffmpeg()
            if cap is not None:
                self.backend = "ffmpeg"
                return cap
        # local file / webcam
        cap = cv2.VideoCapture(self.src)
        if cap.isOpened():
            self.backend = "other"
            return cap
        return None

    def start(self):
        self.cap = self._open()
        if self.cap is None:
            print("[ERRO] Não foi possível abrir o stream.")
            return self
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        # descarta alguns frames iniciais
        for _ in range(3):
            if not self.running:
                return
            self.cap.read()

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            # guarda sempre o mais novo
            with self.lock:
                self.frame = frame

            # Se for FFMPEG/RTSP, tenta drenar um pouquinho o buffer
            if self.backend == "ffmpeg" and isinstance(self.src, str) and self.src.lower().startswith("rtsp"):
                # 2 grabs extras costumam ajudar sem detonar CPU
                self.cap.grab()
                self.cap.grab()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=1)
        except Exception:
            pass
        if self.cap:
            self.cap.release()
