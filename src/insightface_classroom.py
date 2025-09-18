
import os
os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;1024"
)

import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import streamlit as st
import pandas as pd
from datetime import timedelta
import datetime
from control_database import insert_count_behavior, df_behavior_charts, show_behavior_charts
from register_face_multi_images_avg import load_insightface_data
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from insightface.app import FaceAnalysis
import warnings
import hashlib
from utils_criptografia import salvar_mapeamento
from socket_video_stream import VideoStream  # cliente do relay via socket
import threading
from collections import deque

warnings.filterwarnings("ignore", category=FutureWarning)

# Suavização de "Dormindo"
sleep_smoother = {}
ENTER_SLEEP_FRAMES = 6  
EXIT_SLEEP_FRAMES  = 10  

# Configuração da câmera
CAM_SETUP = 'LEFT'
CAM_YAW_OFFSET = 12.0
YAW_LATERAL_THRESH = 28.0


# Paths
DATA_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATABASE_PATH  = os.path.join(DATA_DIR, "alunos")                 # pasta dos alunos
MAPPING_CSV    = os.path.join(DATA_DIR, "mapeamento_alunos.csv")  # pasta mapeamento alunos'

# Imagens da UI
image_path_classroom = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))
image_path_faces     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/faces.png"))
image_path_cam       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/cam_IA.png"))
image_path_table     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/table.png"))

lateral_timers = {}

# ---------------- Associação por IoU + memória curta de nome ----------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)

NAME_TTL = 8.0  # segura o nome por N segundos quando a face some
_name_mem = deque(maxlen=80)

def remember_name(box, name):
    _name_mem.append({"box": box, "name": name, "ts": time.time()})

def resolve_name(person_box):
    now = time.time()
    best, who = 0.0, "Desconhecido"
    for item in list(_name_mem):
        if now - item["ts"] > NAME_TTL:
            continue
        i = iou(person_box, item["box"])
        if i > best:
            best, who = i, item["name"]
    return who if best > 0.03 else "Desconhecido"

# ---------------- Detector em thread separada (IA fora do loop de render) ----------------
class DetectorWorker:
    """
    Roda YOLO (pose) + InsightFace em background, sempre no frame mais recente.
    Evita fila e mantém o vídeo "ao vivo".
    """
    def __init__(self, model_pose, model_face, device):
        self.model_pose = model_pose
        self.model_face = model_face
        self.device = device
        self._latest_frame = None
        self._last_results = []
        self._last_faces = []
        self._lock = threading.Lock()
        self._running = False
        self._th = None

    def start(self):
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._running = False
        try:
            if self._th:
                self._th.join(timeout=1.0)
        except:
            pass

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_outputs(self):
        with self._lock:
            faces = self._last_faces
            results = self._last_results
        return results, faces

    def _run(self):
        while self._running:
            frame = None
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.003)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.model_face.get(rgb)
            results = self.model_pose.predict(frame, show=False, device=self.device, verbose=False, imgsz=896, conf=0.35, half=(self.device == "cuda"))
            with self._lock:
                self._last_faces = faces
                self._last_results = results

# ---------------- Funções auxiliares de comportamento ----------------
def is_lateral_view(nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=0.5, cam_side="LEFT", cam_offset=0.12):
   
    # escala da pessoa
    s = float(abs(ls[0] - rs[0])) + 1e-6

    # limiar dinâmico p/ razão de deslocamento (nariz vs olhos)
    if s < 40:
        t_ratio = 0.22
    elif s < 60:
        t_ratio = 0.28
    else:
        t_ratio = 0.34

    # offset conforme o lado da camera

    if cam_side == "LEFT":
        offset = -cam_offset
    if cam_side == "RIGHT":
        offset = +cam_offset
    else:
        offset = 0.0

    

    # --- OLHOS ---
    eye_dx = float(abs(l_eye[0] - r_eye[0]))
    cond_ratio_eyes = False

    if eye_dx >= 1.0:
        ratio_eyes = abs(nose[0] - (l_eye[0] + r_eye[0]) / 2.0) / eye_dx
        ratio_eyes_corr = ratio_eyes + offset
        cond_ratio_eyes = abs(ratio_eyes_corr) > t_ratio

    # assimetria de confiança dos olhos (um muito baixo e outro alto)
    cond_conf_eyes = (
        (l_eye[2] < conf_thr and r_eye[2] > conf_thr + 0.1) or
        (r_eye[2] < conf_thr and l_eye[2] > conf_thr + 0.1)
    )

    # --- ORELHAS (fallback) ---
    cond_ears = False
    # se ao menos uma orelha tem boa confiança, tenta usar
    if (l_ear[2] > conf_thr) or (r_ear[2] > conf_thr):
        ear_dx = float(abs(l_ear[0] - r_ear[0]))
        if ear_dx >= 1.0:
            ratio_ears = abs(nose[0] - (l_ear[0] + r_ear[0]) / 2.0) / ear_dx

            ratio_ears_corr = ratio_ears + offset

            # ligeiramente mais permissivo no fallback
            cond_ratio_ears = ratio_ears_corr > (t_ratio * 0.90)
        else:
            cond_ratio_ears = False

        # assimetria de confiança das orelhas
        cond_conf_ears = (
            (l_ear[2] < conf_thr and r_ear[2] > conf_thr + 0.1) or
            (r_ear[2] < conf_thr and l_ear[2] > conf_thr + 0.1)
        )
        cond_ears = cond_ratio_ears or cond_conf_ears

    return cond_ratio_eyes or cond_conf_eyes or cond_ears

def check_distracted_status(name, is_lateral, lateral_timers, timeout=10):
    now = time.time()
    if name not in lateral_timers:
        lateral_timers[name] = {"start_time": None, "is_lateral": False}
    if is_lateral:
        if not lateral_timers[name]["is_lateral"] and lateral_timers[name]["start_time"] is None:
            lateral_timers[name]["start_time"] = now
            lateral_timers[name]["is_lateral"] = True
        else:
            elapsed = now - (lateral_timers[name]["start_time"] or now)
            if elapsed >= timeout:
                return "Distraido"
    else:
        lateral_timers[name]["start_time"] = None
        lateral_timers[name]["is_lateral"] = False
    return None

def classify_behavior(nose, ls, rs, le, re, lw, rw, threshold):
    """
    0:nose | 5-6: ombros (ls, rs) | 7-8: cotovelos (le, re) | 9-10: punhos (lw, rw)
    """
    cx = (ls[0] + rs[0]) / 2.0
    cy = (ls[1] + rs[1]) / 2.0
    s  = max(1.0, float(abs(ls[0] - rs[0])))   # escala ombro-a-ombro

    # --- MÃOS ALTAS -> Perguntando/Agitado (robusto à distância) ---
    up_L = (lw[1] < nose[1]) or (lw[1] < (cy - 0.14 * s))
    up_R = (rw[1] < nose[1]) or (rw[1] < (cy - 0.14 * s))
    if up_L and up_R:
        return "Agitado" if abs(lw[0] - rw[0]) > 0.90 * s else "Perguntando"
    if up_L or up_R:
        return "Perguntando"

    # --- DORMINDO: cabeça baixa OU nariz próximo do cotovelo (apoio no braço) ---
    MIN_S_FOR_SLEEP = 28.0
    DY_COEF   = 0.14
    DX_COEF   = 0.35 if s >= 50 else 0.55   # tolera cabeça de lado se estiver longe
    ELB_NEAR  = 0.26

    dy = nose[1] - cy
    dx = abs(nose[0] - cx)
    best_elbow = min(abs(nose[1] - le[1]), abs(nose[1] - re[1]))
    hands_low  = (lw[1] > cy - 0.08 * s) and (rw[1] > cy - 0.08 * s)
    elbows_low = (le[1] > cy - 0.12 * s) and (re[1] > cy - 0.12 * s)

    if s >= MIN_S_FOR_SLEEP and hands_low and elbows_low:
        head_low_ok   = (dy > DY_COEF * s) and (dx < DX_COEF * s)
        elbow_near_ok = (best_elbow < ELB_NEAR * s) and (nose[1] > cy - 0.10 * s)
        if head_low_ok or elbow_near_ok:
            return "Dormindo"

    # --- fallback ---
    return "Atento"


# ------------------ CRIPTOGRAFAR NOMES ---------------------------
def criptografar_nome_matricula(nome, matricula):
    return hashlib.sha256(f"{nome}_{matricula}".encode()).hexdigest()

# ------------------------------ APP ------------------------------
def recognition_behavior():
    school = "Escola Estadual Criança Esperança"
    discipline = "Matemática"

    st.sidebar.image(image_path_classroom, use_container_width=True)
    user_name = st.session_state.get("name", "Usuário")
    st.sidebar.markdown(f"**{user_name}**")

    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.rerun()

    menu_option = st.sidebar.radio("Menu", ["Cadastro de Alunos", "Monitoramento", "Gráficos", "Tabela"])

    # ------------------ CADASTRO ------------------
    if menu_option == "Cadastro de Alunos":
        st.title("📸 Cadastro de Alunos")

        # Parâmetros da captura automática
        IMAGENS_POR_POSE = st.number_input("Imagens por pose", 1, 30, 10, 1)
        capture_interval = st.slider("Intervalo entre fotos (segundos)", 0.2, 3.0, 0.8, 0.1)
        prep_seconds     = st.slider("Contagem inicial (segundos)", 0, 5, 2, 1)

        POSES = ["frontal", "lateral_direita", "lateral_esquerda", "cabeca_baixa"]

        # Estado
        pose_index        = st.session_state.get("pose_index", 0)
        img_index         = st.session_state.get("img_index", 0)
        cap_running       = st.session_state.get("cap_running", False)
        next_time         = st.session_state.get("next_time", None)
        pose_done         = st.session_state.get("pose_done", False)
        registration_done = st.session_state.get("registration_done", False)

        # Tela de conclusão
        if registration_done:
            ultimo_nome = st.session_state.get("last_cad_nome", "")
            ultima_mat  = st.session_state.get("last_cad_matricula", "")
            if ultimo_nome or ultima_mat:
                st.success(f"✅ Cadastro concluído para **{ultimo_nome}** (Matrícula **{ultima_mat}**).")
            else:
                st.success("✅ Cadastro concluído.")
            if st.button("✅ Finalizar cadastro"):
                if 'cadastro_cap' in st.session_state:
                    try: st.session_state.cadastro_cap.release()
                    except: pass
                    del st.session_state['cadastro_cap']

                for k in ["pose_index","img_index","cap_running","next_time","pose_done",
                          "registration_done","last_cad_nome","last_cad_matricula"]:
                    st.session_state.pop(k, None)

                st.session_state.cad_nome = ""
                st.session_state.cad_matricula = ""

                st.toast("Cadastro finalizado.")
                st.rerun()
            st.stop()

        # Entradas
        disciplinas = ["Matemática", "Português", "História", "Geografia", "Ciências"]
        _ = st.selectbox("📘 Selecione a Disciplina:", disciplinas)
        nome_aluno  = st.text_input("Nome do Aluno:", key="cad_nome")
        matricula   = st.text_input("Matrícula do Aluno:", key="cad_matricula")

        if nome_aluno and matricula:
            nome_criptografado = salvar_mapeamento(nome_aluno, matricula)

            os.makedirs(DATABASE_PATH, exist_ok=True)
            pasta_base = os.path.join(DATABASE_PATH, nome_criptografado)
            os.makedirs(pasta_base, exist_ok=True)
            for _pose in POSES:
                os.makedirs(os.path.join(pasta_base, _pose), exist_ok=True)

            # CSV fora da pasta 'alunos'
            if not os.path.exists(MAPPING_CSV):
                pd.DataFrame(columns=["nome", "matricula", "hash"]).to_csv(MAPPING_CSV, index=False)

            df = pd.read_csv(MAPPING_CSV)
            nome_norm = str(nome_aluno).strip()
            matr_norm = str(matricula).strip()
            mask = (df["nome"].astype(str).str.strip() == nome_norm) & \
                   (df["matricula"].astype(str).str.strip() == matr_norm)
            if mask.any():
                df.loc[mask, "hash"] = nome_criptografado
            else:
                df = pd.concat(
                    [df, pd.DataFrame([{"nome": nome_norm, "matricula": matr_norm, "hash": nome_criptografado}])],
                    ignore_index=True
                )
            df = df.drop_duplicates(subset=["nome", "matricula"], keep="first")
            df.to_csv(MAPPING_CSV, index=False)

            pose_index = max(0, min(pose_index, len(POSES) - 1))
            pose_atual = POSES[pose_index]
            st.subheader(f"👉 Pose atual: **{pose_atual.replace('_',' ').title()}**  ({img_index}/{IMAGENS_POR_POSE})")

            # Preview
            stframe = st.empty()
            if 'cadastro_cap' not in st.session_state:
                st.session_state.cadastro_cap = cv2.VideoCapture(0)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap = st.session_state.cadastro_cap

            cols = st.columns(3)
            with cols[0]:
                start_btn = st.button("▶️ Iniciar captura desta pose", disabled=cap_running or pose_done)
            with cols[1]:
                cancel_btn = st.button("⏹️ Cancelar captura", disabled=not cap_running)
            with cols[2]:
                next_btn = st.button("➡️ Próximo",
                    disabled=st.session_state.get("cap_running", False) or not st.session_state.get("pose_done", False))

            if start_btn:
                st.session_state.cap_running = True
                st.session_state.pose_done   = False
                st.session_state.img_index   = 0
                st.session_state.next_time   = time.time() + prep_seconds
                cap_running = True
                img_index   = 0
                next_time   = st.session_state.next_time

            if cancel_btn:
                st.session_state.cap_running = False
                cap_running = False

            if next_btn and pose_done:
                if (pose_index + 1) < len(POSES):
                    st.session_state.pose_index  = (pose_index + 1)
                    st.session_state.img_index   = 0
                    st.session_state.pose_done   = False
                    st.session_state.cap_running = False
                    st.rerun()
                else:
                    st.session_state.registration_done = True
                    st.session_state.last_cad_nome = nome_norm
                    st.session_state.last_cad_matricula = matr_norm
                    st.session_state.cap_running = False
                    st.session_state.pose_done   = False
                    st.session_state.next_time   = None
                    st.rerun()

            if cap_running:
                pasta_pose = os.path.join(pasta_base, pose_atual)
                os.makedirs(pasta_pose, exist_ok=True)
                while st.session_state.cap_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Não foi possível ler da câmera.")
                        break
                    now = time.time()
                    restante = max(0.0, (st.session_state.next_time or now) - now)
                    overlay = frame.copy()
                    cv2.putText(overlay, f"Pose: {pose_atual.replace('_',' ').title()}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(overlay, f"Foto: {st.session_state.img_index}/{IMAGENS_POR_POSE}",
                                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(overlay, f"Proxima em: {restante:0.1f}s",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    stframe.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", width=480)

                    if now >= (st.session_state.next_time or now):
                        timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        nome_arquivo = f"{pose_atual}_{timestamp}.jpg"
                        caminho      = os.path.join(pasta_pose, nome_arquivo)
                        cv2.imwrite(caminho, frame)
                        st.session_state.img_index += 1
                        st.session_state.next_time  = now + capture_interval

                        if st.session_state.img_index >= IMAGENS_POR_POSE:
                            st.session_state.cap_running = False
                            st.session_state.pose_done   = True
                            st.session_state.next_time   = None

                            if pose_index == len(POSES) - 1:
                                st.session_state.registration_done = True
                                st.session_state.last_cad_nome = nome_norm
                                st.session_state.last_cad_matricula = matr_norm
                                st.rerun()
                            else:
                                st.success(
                                    f"✅ {IMAGENS_POR_POSE} imagens capturadas para '{pose_atual}'. "
                                    f"Clique em **Próximo** para a próxima pose."
                                )
                                st.rerun()
                    time.sleep(0.02)
            else:
                ret, frame = cap.read()
                if ret:
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=480)
        else:
            st.warning("Preencha a disciplina, nome e matrícula do aluno para iniciar a captura.")

    # ------------------ MONITORAMENTO ------------------
    elif menu_option == "Monitoramento":
        # Fecha webcam de cadastro se aberta
        if 'cadastro_cap' in st.session_state:
            try: st.session_state.cadastro_cap.release()
            except: pass
            del st.session_state['cadastro_cap']

        # Fecha stream antigo (evita múltiplos leitores após rerun)
        if 'video_stream' in st.session_state:
            try: st.session_state.video_stream.stop()
            except: pass
            del st.session_state['video_stream']

        col_img1, col_img2, _ = st.columns([1,4,1])
        with col_img1:
            st.image(image_path_cam, width=200)
        with col_img2:
            st.title("MONITORAMENTO")

        CONFIDENCE_THRESHOLD = st.sidebar.slider("Confiança Mínima", 0.10, 0.80, 0.35, 0.05)
        use_gpu = st.sidebar.checkbox("Usar GPU (CUDA)", value=True)
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        st.sidebar.write(f"Dispositivo: {device}")

        # HUD de debug no canto esquerdo
        show_debug = st.sidebar.toggle("Mostrar debug (Dormindo)", value=False)
        debug_font = st.sidebar.slider("Tamanho fonte debug", 0.4, 2.0, 0.8, 0.1)

        col1, col2 = st.sidebar.columns(2)
        run_system = col1.button("Iniciar Monitoramento")
        stop_system = col2.button("Parar Monitoramento")

        model = YOLO('yolo11m-pose.pt')
        behavior_tracker = {}
        BOX_MARGIN_RATIO = 0.2

        # FaceAnalysis
        if device == "cuda":
            model_face = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            model_face.prepare(ctx_id=0, det_size=(832,832))
        else:
            model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            model_face.prepare(ctx_id=-1, det_size=(832,832))

        # Embeddings conhecidos + normalização fora do loop
        known_face_encodings, known_face_names = load_insightface_data()
        known_face_encodings_norm = (
            known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
        ) if len(known_face_encodings) > 0 else None

        messege = st.empty()
        if not run_system and not stop_system:
            messege.info("Obs: O sistema irá monitorar os comportamentos dos alunos durante a aula. Inicie o monitoramento!")

        if run_system:
            messege.empty()
            stframe = st.empty()

            fps_limit = 20
            prev_time = 0.0

            video_stream = VideoStream(("172.16.5.158", 5555)).start()
            st.session_state.video_stream = video_stream

            # flush rápido para pegar frame atual
            t0 = time.time()
            while time.time() - t0 < 0.3:
                _ = video_stream.read()

            detector = DetectorWorker(model, model_face, device).start()

            while video_stream.running:
                if time.time() - prev_time < 1.0 / fps_limit:
                    time.sleep(0.001)
                    continue
                prev_time = time.time()

                frame = video_stream.read()
                if frame is None:
                    continue

                detector.update_frame(frame)
                results, faces = detector.get_outputs()

                # ---- FACES (nomeia e guarda na memória curto prazo) ----
                face_named = []  # lista de ((fx1,fy1,fx2,fy2), nome)
                if faces:
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                        name_face = "Desconhecido"
                        if known_face_encodings_norm is not None:
                            emb = face.embedding
                            emb = emb / (np.linalg.norm(emb) + 1e-6)
                            sims = cosine_similarity([emb], known_face_encodings_norm)[0]
                            best_idx = int(np.argmax(sims))
                            if float(sims[best_idx]) > 0.45:
                                name_face = known_face_names[best_idx]
                        face_named.append(((fx1, fy1, fx2, fy2), name_face))
                        if name_face != "Desconhecido":
                            remember_name((fx1, fy1, fx2, fy2), name_face)

                # ---- POSE + Lógica de comportamento ----
                if results:
                    for result in results:
                        if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
                            continue
                        keypoints_all = result.keypoints.data.cpu().numpy()

                        # desenhar HUD no canto esquerdo (uma vez por frame; atualiza com a última pessoa válida)
                        hud_lines = []

                        for pid, person_keypoints in enumerate(keypoints_all):
                            if len(person_keypoints) == 0:
                                continue

                            current_behavior = "Atento"
                            have_all = False

                            if person_keypoints.shape[0] > 10:
                                nose = person_keypoints[0]
                                ls, rs = person_keypoints[5], person_keypoints[6]   # OMBROS
                                le, re = person_keypoints[7], person_keypoints[8]   # COTOVELOS
                                lw, rw = person_keypoints[9], person_keypoints[10]  # PUNHOS

                                confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
                                have_all = all(c > CONFIDENCE_THRESHOLD for c in confs)
                                if have_all:
                                    current_behavior = classify_behavior(nose, ls, rs, le, re, lw, rw, CONFIDENCE_THRESHOLD)

                            # Caixa da pessoa pelos keypoints
                            x_coords = [p[0] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                            y_coords = [p[1] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                            if not x_coords or not y_coords:
                                continue
                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))
                            y_min = max(0, int(y_min - BOX_MARGIN_RATIO * (y_max - y_min)))
                            person_box = (x_min, y_min, x_max, y_max)

                            # 1) tenta associar pelo IoU com faces do frame
                            best_i, name_student = 0.0, "Desconhecido"
                            for (fb, nm) in face_named:
                                i = iou(person_box, fb)
                                if i > best_i:
                                    best_i, name_student = i, nm
                            # 2) se não achou, tenta memória curta
                            if best_i < 0.10:
                                name_student = resolve_name(person_box)

                            # Distraído (só quando tem nome válido)
                            if name_student != "Desconhecido" and person_keypoints.shape[0] > 10:
                                nose  = person_keypoints[0]
                                l_eye = person_keypoints[1]   # olho E
                                r_eye = person_keypoints[2]   # olho D
                                l_ear = person_keypoints[3]   # orelha E
                                r_ear = person_keypoints[4]   # orelha D
                                ls    = person_keypoints[5]   # ombro E
                                rs    = person_keypoints[6]   # ombro D

                                # só roda se temos nariz + ao menos um par (olhos ou orelhas) usável
                                if nose[2] > CONFIDENCE_THRESHOLD:
                                    lateral_status = is_lateral_view(
                                        nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=CONFIDENCE_THRESHOLD
                                    )
                                    new_behavior = check_distracted_status(
                                        name_student, lateral_status, lateral_timers, timeout=10
                                    )
                                    if new_behavior:
                                        current_behavior = new_behavior


                            # =================== HISTERese Dormindo <-> Atento ===================
                                raw_behavior = current_behavior
                                key = name_student if name_student != "Desconhecido" else f"pid_{pid}"
                                state = sleep_smoother.setdefault(key, {"state":"Atento","sleep":0,"awake":0})

                                if raw_behavior == "Dormindo":
                                    state["sleep"] += 1; state["awake"] = 0
                                    if state["state"] != "Dormindo" and state["sleep"] >= ENTER_SLEEP_FRAMES:
                                        state["state"] = "Dormindo"
                                else:
                                    state["awake"] += 1; state["sleep"] = 0
                                    if state["state"] == "Dormindo" and state["awake"] >= EXIT_SLEEP_FRAMES:
                                        state["state"] = raw_behavior
                                    elif state["state"] != "Dormindo":
                                        state["state"] = raw_behavior

                                current_behavior = state["state"]
                                # =================== FIM DA HISTERese ============================================



                            # Registro no DB (transições)
                            date = datetime.datetime.now().strftime("%Y-%m-%d")
                            current_time = datetime.datetime.now().strftime("%H:%M:%S")

                            if name_student not in behavior_tracker:
                                behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                            if behavior_tracker[name_student]["behavior"] != current_behavior and name_student != "Desconhecido":
                                insert_count_behavior(
                                    school, discipline, user_name, '12345', name_student,
                                    behavior_tracker[name_student]["behavior"], date,
                                    behavior_tracker[name_student]["start_time"], current_time
                                )
                                behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                            # ------------------- Desenho da caixa/label (com cor por comportamento) -------------------
                            # BGR: vermelho (0,0,255) para Agitado/Dormindo; verde (0,255,0) para os demais

                            box_color = (0, 0, 255) if current_behavior in ("Agitado", "Dormindo", "Distraido") else (0, 255, 0)

                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                            
                            label_text =  f"{name_student} - {current_behavior}"

                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.6
                            thickness = 2
                            pad_x, pad_y = 6, 4

                            # Tamanho do texto
                            (text_w, text_h), _ = cv2.getTextSize(label_text, font, scale, thickness)
                            

                            # Posição: canto superior esquerdo da box
                            tx = int(x_min)
                            ty = int(y_min)

                            top = ty - text_h - 2*pad_y
                            if top < 0:
                                top = ty
                                ty = top + text_h + pad_y

                            cv2.rectangle(frame, (tx, top), (tx + text_w + 2*pad_x, top + text_h + 2*pad_y), box_color, -1) # Fundo da cor da box
                            
                            cv2.putText(frame, label_text, (tx + pad_x, top + text_h + pad_y - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA) # Texto branco por cima
                             

                            # -------- HUD de debug no canto esquerdo --------
                            if show_debug and have_all:
                                # Recalcular variáveis para exibir (mesma lógica da função):
                                shoulder_y = (ls[1] + rs[1]) / 2.0
                                s = max(1.0, abs(ls[0] - rs[0]))
                                best_vert_dist = min(abs(nose[1] - le[1]), abs(nose[1] - re[1]))
                                near_thr = max(10.0, 0.32 * s)

                                hud_lines = [
                                    f"s (ombro a ombro): {s:.1f}",
                                    f"near_thr: {near_thr:.1f}",
                                    f"shoulder_y: {shoulder_y:.1f}",
                                    f"nose_y: {nose[1]:.1f}",
                                    f"best_vert_dist: {best_vert_dist:.1f}",
                                ]
                                # desenha do lado esquerdo
                                y0 = 24
                                for i, text in enumerate(hud_lines):
                                    cv2.putText(frame, text, (10, y0 + int(i * 22 * debug_font)),
                                                
                                                cv2.FONT_HERSHEY_SIMPLEX, debug_font, (255, 255, 0), 2)
                                    
                            
                            
                # Render leve
                disp = cv2.resize(frame, (960, 540))
                stframe.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # encerra ao sair
            try: detector.stop()
            except: pass
            try: video_stream.stop()
            except: pass
            if 'video_stream' in st.session_state:
                del st.session_state['video_stream']

        if stop_system:
            st.info("Monitoramento parado.")
            if 'video_stream' in st.session_state:
                try: st.session_state.video_stream.stop()
                except: pass
                del st.session_state['video_stream']

    # ------------------ GRÁFICOS ------------------
    elif menu_option == "Gráficos":
        st.title("📊 GRÁFICOS")
        show_behavior_charts()

    # ------------------ TABELA ------------------
    elif menu_option == "Tabela":
        col_img1, col_img2, _ = st.columns([1, 6, 1])
        with col_img1:
            st.image(image_path_table, width=200)
        with col_img2:
            st.title("INFORMAÇÕES")

        df = df_behavior_charts()
        if df.empty:
            st.warning("Nenhum dado registrado.")
            return

        today = datetime.datetime.now()
        selected_date = st.date_input("Selecione a Data", value=today,
                                       min_value=today - timedelta(days=365),
                                       max_value=today + timedelta(days=365))

        selected_disciplines = st.multiselect("Filtrar por Disciplinas", df['Disciplina'].unique().tolist())
        selected_behaviors = st.multiselect("Filtrar por Comportamentos", df['Comportamento'].unique().tolist())

        df['Data'] = pd.to_datetime(df['Data']).dt.date
        filtered_df = df[df['Data'] == selected_date]

        if selected_disciplines:
            filtered_df = filtered_df[filtered_df['Disciplina'].isin(selected_disciplines)]
        if selected_behaviors:
            filtered_df = filtered_df[filtered_df['Comportamento'].isin(selected_behaviors)]

        if filtered_df.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            st.dataframe(filtered_df, use_container_width=True)