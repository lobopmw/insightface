# socket_video_stream.py
import socket, struct, threading, time
import cv2, numpy as np

class VideoStream:
    """
    Cliente de frames via socket (mesma API: start()/read()/stop()).
    Conecta no relay: (host, porta), ex.: ("127.0.0.1", 5555)
    """
    def __init__(self, server=("127.0.0.1", 5555), reconnect_sec=2):
        self.server = server
        self.reconnect_sec = reconnect_sec
        self.sock = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect(self.server)
        self.sock = s

    def _recvall(self, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except Exception:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _reader(self):
        while self.running:
            try:
                if self.sock is None:
                    self._connect()
                hdr = self._recvall(4)
                if not hdr: raise RuntimeError("header")
                (size,) = struct.unpack(">I", hdr)
                data = self._recvall(size)
                if not data: raise RuntimeError("payload")
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self.lock:
                    self.frame = frame
            except Exception:
                try:
                    if self.sock: self.sock.close()
                except Exception:
                    pass
                self.sock = None
                time.sleep(self.reconnect_sec)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            if self.thread: self.thread.join(timeout=1)
        except Exception:
            pass
        try:
            if self.sock: self.sock.close()
        except Exception:
            pass
