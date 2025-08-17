# relay_rtsp_server.py
# Servidor que lê RTSP (Hikvision) e envia SEMPRE o frame mais novo via TCP.
# Agora envia SOMENTE quando chega um frame novo e pode limitar com --send-fps.
# Uso:
#   python relay_rtsp_server.py "rtsp://admin:admin123@172.16.5.250:554/Streaming/Channels/101" --host 0.0.0.0 --port 5555 --quality 85 --send-fps 15

import os
# Defina as opções ANTES do import cv2 (baixa latência no FFMPEG)
os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;1024"

import cv2
import socket
import struct
import threading
import argparse
import time

def open_capture(rtsp_url: str, width=None, height=None):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # força FFMPEG
    if not cap.isOpened():
        raise RuntimeError("Não abriu o RTSP com FFMPEG. Verifique URL/credenciais/rede.")
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # pedir buffer mínimo
    except Exception: pass
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def start_relay(rtsp_url: str, host="0.0.0.0", port=5555, quality=85, width=None, height=None, send_fps=None):
    cap = open_capture(rtsp_url, width, height)

    last_frame = [None]
    last_id = [0]
    lock = threading.Lock()
    running = True

    def grabber():
        # descarta alguns frames iniciais
        for _ in range(3): cap.read()
        while running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with lock:
                last_frame[0] = frame
                last_id[0] += 1  # marca "chegou frame novo"

    def get_latest_if_new(prev_id):
        with lock:
            if last_id[0] == prev_id or last_frame[0] is None:
                return prev_id, None
            return last_id[0], last_frame[0].copy()

    def client_worker(conn):
        try:
            sent_id = -1
            interval = (1.0 / float(send_fps)) if send_fps and send_fps > 0 else 0.0
            next_deadline = 0.0
            while True:
                # se limitar FPS, respeite o intervalo
                if interval and time.time() < next_deadline:
                    time.sleep(0.001)
                    continue

                # só envia se houver frame NOVO
                sent_id, frame = get_latest_if_new(sent_id)
                if frame is None:
                    time.sleep(0.001)
                    continue

                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    continue
                data = jpg.tobytes()
                header = struct.pack(">I", len(data))
                conn.sendall(header + data)

                if interval:
                    next_deadline = time.time() + interval
        except Exception:
            pass
        finally:
            try: conn.close()
            except: pass

    t = threading.Thread(target=grabber, daemon=True)
    t.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(2)
    print(f"[relay] Servindo frames em tcp://{host}:{port}")
    print(f"[relay] Capturando de: {rtsp_url}")
    if send_fps:
        print(f"[relay] Limitando envio a ~{send_fps} fps")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[relay] cliente conectado: {addr}")
            threading.Thread(target=client_worker, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        try: srv.close()
        except: pass
        try:
            running = False
            t.join(timeout=1.0)
        except: pass
        try: cap.release()
        except: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtsp", help="URL RTSP (ex.: rtsp://user:pass@IP:554/Streaming/Channels/101)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--quality", type=int, default=85, help="Qualidade JPEG (50–95). 85 = bom e rápido")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--send-fps", type=int, default=None, help="Limitar FPS de envio (ex.: 15, 20, 25). Se omitido, envia o mais rápido possível, porém só quando chega frame novo.")
    args = ap.parse_args()
    start_relay(args.rtsp, host=args.host, port=args.port, quality=args.quality,
                width=args.width, height=args.height, send_fps=args.send_fps)

if __name__ == "__main__":
    main()
