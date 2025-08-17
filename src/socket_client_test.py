# socket_client_test.py (headless)
import socket, struct, argparse, cv2, numpy as np, time

def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--save_every", type=float, default=1.0, help="segundos entre salvamentos de preview.jpg")
    args = ap.parse_args()

    addr = (args.host, args.port)
    print(f"[client] conectando em tcp://{addr[0]}:{addr[1]} ...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect(addr)
    print("[client] conectado. Ctrl+C para sair.")

    last_save = 0.0
    t0 = time.time()
    frames = 0

    try:
        while True:
            hdr = recvall(s, 4)
            if hdr is None:
                print("[client] conexão encerrada (header)."); break
            (size,) = struct.unpack(">I", hdr)
            data = recvall(s, size)
            if data is None:
                print("[client] conexão encerrada (payload)."); break

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # métricas simples
            frames += 1
            now = time.time()
            if now - t0 >= 1.0:
                print(f"[client] fps ~ {frames}/s   frame={frame.shape[1]}x{frame.shape[0]}")
                frames = 0
                t0 = now

            # salva um preview periodicamente (para ver se está chegando imagem)
            if now - last_save >= args.save_every:
                cv2.imwrite("preview.jpg", frame)
                last_save = now

    except KeyboardInterrupt:
        pass
    finally:
        try: s.close()
        except: pass
        print("[client] saindo.")

if __name__ == "__main__":
    main()
