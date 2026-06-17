# socket_video_stream.py
import select
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
        self.frame_jpeg = None
        self.lock = threading.Lock()
        self.thread = None
        self.connected = False
        self.last_error = None
        self.last_frame_at = None
        self.frames_received = 0
        self.frame_id = 0
        self._decoded_frame_id = -1
        self._socket_timeout_sec = 1.0
        self._recv_buffer = bytearray()
        self._max_payload_size = 8 * 1024 * 1024

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        s.settimeout(self._socket_timeout_sec)
        s.connect(self.server)
        self.sock = s
        self.connected = True
        self.last_error = None

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

    def _recv_into_buffer(self):
        if self.sock is None:
            return False

        received_any = False
        while True:
            try:
                ready, _, _ = select.select([self.sock], [], [], 0)
            except Exception:
                ready = []
            if not ready:
                break
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except Exception:
                return False
            if not chunk:
                return False
            self._recv_buffer.extend(chunk)
            received_any = True
        return received_any

    def _extract_latest_packet(self):
        latest_payload = None
        while len(self._recv_buffer) >= 4:
            size = struct.unpack(">I", self._recv_buffer[:4])[0]
            if size <= 0 or size > self._max_payload_size:
                raise RuntimeError(f"payload_size_invalido:{size}")
            total_size = 4 + size
            if len(self._recv_buffer) < total_size:
                break
            latest_payload = bytes(self._recv_buffer[4:total_size])
            del self._recv_buffer[:total_size]
        return latest_payload

    def _reader(self):
        while self.running:
            try:
                if self.sock is None:
                    self._connect()

                if len(self._recv_buffer) < 4:
                    hdr = self._recvall(4)
                    if not hdr:
                        raise RuntimeError("header")
                    self._recv_buffer.extend(hdr)

                size = struct.unpack(">I", self._recv_buffer[:4])[0]
                if size <= 0 or size > self._max_payload_size:
                    raise RuntimeError(f"payload_size_invalido:{size}")

                missing = (4 + size) - len(self._recv_buffer)
                if missing > 0:
                    data = self._recvall(missing)
                    if not data:
                        raise RuntimeError("payload")
                    self._recv_buffer.extend(data)

                self._recv_into_buffer()
                latest_payload = self._extract_latest_packet()
                if latest_payload is None:
                    raise RuntimeError("payload")

                with self.lock:
                    self.frame_jpeg = latest_payload
                    self.frame = None
                    self._decoded_frame_id = -1
                    self.last_frame_at = time.time()
                    self.frames_received += 1
                    self.frame_id += 1
            except Exception as exc:
                self.connected = False
                self.last_error = repr(exc)
                try:
                    if self.sock: self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self._recv_buffer.clear()
                time.sleep(self.reconnect_sec)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def read(self):
        frame, _, _, _ = self.read_with_meta()
        return frame

    def read_with_meta(self):
        with self.lock:
            frame_id = self.frame_id
            last_frame_at = self.last_frame_at
            jpeg = self.frame_jpeg
            cached_frame = self.frame
            decoded_frame_id = self._decoded_frame_id

        if jpeg is None:
            return None, frame_id, last_frame_at

        frame = cached_frame
        if frame is None or decoded_frame_id != frame_id:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                return None, frame_id, last_frame_at
            frame = decoded
            with self.lock:
                if self.frame_id == frame_id:
                    self.frame = frame
                    self._decoded_frame_id = frame_id

        return frame.copy(), frame_id, last_frame_at

    def read_jpeg_with_meta(self):
        with self.lock:
            if self.frame_jpeg is None:
                return None, self.frame_id, self.last_frame_at
            return self.frame_jpeg, self.frame_id, self.last_frame_at

    def read_latest_with_meta(self):
        with self.lock:
            frame_id = self.frame_id
            last_frame_at = self.last_frame_at
            jpeg = self.frame_jpeg
            cached_frame = self.frame
            decoded_frame_id = self._decoded_frame_id

        if jpeg is None:
            return None, None, frame_id, last_frame_at

        frame = cached_frame
        if frame is None or decoded_frame_id != frame_id:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                return None, None, frame_id, last_frame_at
            frame = decoded
            with self.lock:
                if self.frame_id == frame_id:
                    self.frame = frame
                    self._decoded_frame_id = frame_id

        return frame.copy(), bytes(jpeg), frame_id, last_frame_at

    def stop(self):
        self.running = False
        self.connected = False
        try:
            if self.thread: self.thread.join(timeout=1)
        except Exception:
            pass
        try:
            if self.sock: self.sock.close()
        except Exception:
            pass

    def get_status(self):
        with self.lock:
            return {
                "server": self.server,
                "connected": self.connected,
                "last_error": self.last_error,
                "last_frame_at": self.last_frame_at,
                "frames_received": self.frames_received,
                "has_frame": self.frame is not None,
                "frame_id": self.frame_id,
            }
