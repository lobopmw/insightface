
import base64
import html
import io
import os
import glob
from urllib.parse import quote
os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;1024"
)
import sys

import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import streamlit as st
import pandas as pd
from datetime import timedelta
import datetime
from streamlit_cookies_controller import CookieController
from control_database_postgres import (
    APP_TIMEZONE,
    DEFAULT_SCHOOL_NAME,
    DEFAULT_LESSON_TYPE,
    SESSION_STATUS_OPEN,
    SESSION_STATUS_CLOSED,
    close_monitoring_session,
    create_monitoring_session,
    get_local_now,
    get_monitoring_session_summary,
    get_next_student_registration,
    get_student_lookup_for_scope,
    insert_behavior_episode,
    list_classes_for_user,
    list_students_for_user,
    list_subjects_for_user,
    professor_has_assignment,
    show_behavior_charts,
    upsert_student,
)
from register_face_multi_images_avg import generate_student_embedding, load_insightface_data
from PIL import Image
from insightface.app import FaceAnalysis
import warnings
import hashlib
from utils_criptografia import gerar_hash_nome_matricula
from socket_video_stream import VideoStream  # cliente do relay via socket
import threading
from collections import deque
from behavior_episode_service import BehaviorEpisodeManager
from ui.admin_user_page import render_admin_user_page
from ui.report_page import render_report_page

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    av = None
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = exc

warnings.filterwarnings("ignore", category=FutureWarning)

# Suavização local dos comportamentos.
# "Atento" deixa de ser fallback automático e passa a exigir evidência estável.
sleep_smoother = {}
ENTER_SLEEP_FRAMES = 6
EXIT_SLEEP_FRAMES = 10
ENTER_QUESTION_FRAMES = 4
ENTER_ATTENTIVE_FRAMES = 5
ENTER_AGITATED_FRAMES = 3
ENTER_DISTRACTED_FRAMES = 3
ENTER_UNDETERMINED_FRAMES = 2

# Configuração da câmera
CAM_SETUP = 'LEFT'
CAM_YAW_OFFSET = 12.0
YAW_LATERAL_THRESH = 28.0

RELAY_HOST = os.getenv("RELAY_HOST", "127.0.0.1")
RELAY_PORT = int(os.getenv("RELAY_PORT", "5555"))
RELAY_HTTP_PORT = int(os.getenv("RELAY_HTTP_PORT", "8555"))


# Paths
DATA_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATABASE_PATH  = os.path.join(DATA_DIR, "alunos")                 # pasta dos alunos
MAPPING_CSV    = os.path.join(DATA_DIR, "mapeamento_alunos.csv")  # pasta mapeamento alunos'

# Imagens da UI
image_path_classroom = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))
image_path_faces     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/faces.png"))
image_path_cam       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/cam_IA.png"))
image_path_table     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/table.png"))
AUTH_COOKIE_NAME = "auth_user_cpf"
AUTH_QUERY_TOKEN_KEY = "auth_token"
LEGACY_AUTH_QUERY_KEYS = ("authenticated", "cpf", "city", "state", "name", "role")
AUTH_RESTORE_BLOCK_KEY = "auth_restore_blocked"
AUTH_BOOTSTRAP_KEY = "auth_bootstrap_checked"
AUTH_BOOTSTRAP_STARTED_AT_KEY = "auth_bootstrap_started_at"

lateral_timers = {}
DISTRACTED_TIMEOUT_SECONDS = 2.5
UNKNOWN_IDENTITY_LABELS = {"desconhecido", "unknown", ""}

FACE_DET_SIZE_GPU = (1280, 1280)
FACE_DET_SIZE_CPU = (800, 800)
POSE_IMGSZ_GPU = 1280
POSE_IMGSZ_CPU = 800
POSE_DET_CONF = 0.18
FACE_REFRESH_INTERVAL_GPU = 0.08
FACE_REFRESH_INTERVAL_CPU = 0.24
FACE_RECOGNITION_BASE_THRESHOLD = 0.45
FACE_RECOGNITION_MEDIUM_THRESHOLD = 0.39
FACE_RECOGNITION_SMALL_THRESHOLD = 0.30
FACE_RECOGNITION_TINY_THRESHOLD = 0.22
FACE_RECOGNITION_MIN_MARGIN = 0.015
FACE_RECOGNITION_TINY_MARGIN = 0.0025
FAR_FACE_REGION_TOP_RATIO = float(os.getenv("FAR_FACE_REGION_TOP_RATIO", "0.74"))
FAR_FACE_UPSCALE = float(os.getenv("FAR_FACE_UPSCALE", "1.90"))
FAR_FACE_EXTRA_PASS_MAX_BASE_FACES = int(os.getenv("FAR_FACE_EXTRA_PASS_MAX_BASE_FACES", "6"))
FAR_FACE_MERGE_IOU = float(os.getenv("FAR_FACE_MERGE_IOU", "0.20"))
DISPLAY_BOX_MIN_KEYPOINT_CONF = 0.22
DISPLAY_BOX_MIN_WIDTH = 56.0

CAPTURE_POSE_LABELS = {
    "frontal": "Frontal",
    "lateral_esquerda": "Esquerda",
    "lateral_direita": "Direita",
    "cabeca_baixa": "Cabeça baixa",
}
LESSON_TYPE_OPTIONS = ["Exposição", "Atividade", "Prova", "Revisão", "Outro"]
USE_DIRECT_MONITOR_STREAM = False
REGISTRATION_CAMERA_INDEX = int(os.getenv("CADASTRO_CAMERA_INDEX", os.getenv("CAMERA_INDEX", "0")))
REGISTRATION_CAMERA_WIDTH = int(os.getenv("CADASTRO_CAMERA_WIDTH", "640"))
REGISTRATION_CAMERA_HEIGHT = int(os.getenv("CADASTRO_CAMERA_HEIGHT", "480"))
REGISTRATION_CAMERA_WARMUP_FRAMES = max(3, int(os.getenv("CADASTRO_CAMERA_WARMUP_FRAMES", "8")))
MONITOR_FRAME_STALE_SECONDS = float(os.getenv("MONITOR_FRAME_STALE_SECONDS", "2.5"))
MONITOR_UI_REFRESH_SECONDS = float(os.getenv("MONITOR_UI_REFRESH_SECONDS", "0.28"))
MONITOR_DETECTOR_STALE_SECONDS = float(os.getenv("MONITOR_DETECTOR_STALE_SECONDS", "1.8"))
MONITOR_SYNC_FALLBACK_COOLDOWN = float(os.getenv("MONITOR_SYNC_FALLBACK_COOLDOWN", "0.9"))
MONITOR_RECOGNIZED_HOLD_SECONDS = float(os.getenv("MONITOR_RECOGNIZED_HOLD_SECONDS", "3.5"))


def _release_registration_camera() -> None:
    cap = st.session_state.get("cadastro_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.pop("cadastro_cap", None)
    st.session_state.pop("cadastro_camera_error", None)
    st.session_state.pop("cadastro_camera_index", None)
    st.session_state.pop("cadastro_camera_backend", None)


def _open_registration_camera():
    candidate_indexes = []
    preferred_indexes = [REGISTRATION_CAMERA_INDEX]
    fallback_indexes = []

    for device_path in sorted(glob.glob("/dev/video*")):
        suffix = device_path.replace("/dev/video", "", 1)
        if suffix.isdigit():
            fallback_indexes.append(int(suffix))

    for index in preferred_indexes + fallback_indexes + [0, 1, 2, 3]:
        if index not in candidate_indexes:
            candidate_indexes.append(index)

    backend_candidates = []
    if hasattr(cv2, "CAP_V4L2"):
        backend_candidates.append(("V4L2", cv2.CAP_V4L2))
    backend_candidates.append(("default", cv2.CAP_ANY))

    tried = []
    for camera_index in candidate_indexes:
        for backend_name, backend in backend_candidates:
            tried.append(f"indice {camera_index} ({backend_name})")
            try:
                cap = cv2.VideoCapture(camera_index, backend)
            except Exception:
                cap = None

            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REGISTRATION_CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REGISTRATION_CAMERA_HEIGHT)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            for _ in range(REGISTRATION_CAMERA_WARMUP_FRAMES):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    st.session_state["cadastro_camera_index"] = camera_index
                    st.session_state["cadastro_camera_backend"] = backend_name
                    return cap, None
                time.sleep(0.06)

            try:
                cap.release()
            except Exception:
                pass

    tried_msg = ", ".join(tried) if tried else f"indice {REGISTRATION_CAMERA_INDEX}"
    return None, (
        "Nao foi possivel abrir a webcam do cadastro. "
        f"Tentativas: {tried_msg}. "
        "Verifique se outra aplicacao esta usando a camera, se o container recebeu acesso a `/dev/video*`, "
        "ou ajuste a variavel `CADASTRO_CAMERA_INDEX`."
    )


def _normalize_student_name(value: str) -> str:
    return " ".join((value or "").strip().split())


def _normalize_student_registration(value: str) -> str:
    return "".join((value or "").strip().split())


def _validate_student_registration_fields(name: str, matricula: str, require_matricula: bool = True) -> list[str]:
    errors = []
    normalized_name = _normalize_student_name(name)
    normalized_matricula = _normalize_student_registration(matricula)

    letter_count = sum(char.isalpha() for char in normalized_name)
    digit_count_name = sum(char.isdigit() for char in normalized_name)

    if len(normalized_name) < 3:
        errors.append("Informe o nome completo do aluno com pelo menos 3 caracteres.")
    if letter_count < 2:
        errors.append("O campo nome deve conter letras suficientes para identificar o aluno.")
    if digit_count_name > 0 and letter_count <= digit_count_name:
        errors.append("O campo nome parece conter uma matrícula. Revise os campos antes de continuar.")

    if not require_matricula:
        return errors

    if not normalized_matricula:
        errors.append("Informe a matrícula do aluno.")
    elif len(normalized_matricula) < 3:
        errors.append("A matrícula deve ter pelo menos 3 caracteres.")

    return errors


def _format_student_option(student_row: dict) -> str:
    name = _normalize_student_name(student_row.get("name", ""))
    return name or "Aluno sem nome"


def _decode_browser_capture(uploaded_file):
    if uploaded_file is None:
        return None, None, None

    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        return None, None, None

    image_digest = hashlib.sha256(image_bytes).hexdigest()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_rgb, frame_bgr, image_digest


class StudentRegistrationVideoProcessor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame_bgr = None

    def recv(self, frame):
        if av is None:
            return frame
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._latest_frame_bgr = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_frame_bgr(self):
        with self._lock:
            if self._latest_frame_bgr is None:
                return None
            return self._latest_frame_bgr.copy()


def _inject_student_registration_styles() -> None:
    st.markdown(
        """
        <style>
        .student-reg-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 0.15rem 0 1.15rem 0;
        }
        .student-reg-icon {
            width: 68px;
            height: 68px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.2rem;
            background: linear-gradient(180deg, rgba(90,128,255,0.24) 0%, rgba(55,84,168,0.16) 100%);
            border: 1px solid rgba(128,155,255,0.22);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .student-reg-icon svg {
            width: 38px;
            height: 38px;
            display: block;
        }
        .student-reg-title {
            margin: 0;
            font-size: 2.05rem;
            line-height: 1.08;
            font-weight: 800;
            color: #F5F7FB;
        }
        .student-reg-subtitle {
            margin: 0.28rem 0 0 0;
            color: #A9B1BF;
            font-size: 1rem;
            line-height: 1.55;
        }
        .student-reg-card-title {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.9rem;
        }
        .student-reg-card-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.15rem;
            color: white;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
        }
        .student-reg-card-heading {
            margin: 0;
            font-size: 1.08rem;
            font-weight: 800;
            color: #F5F7FB;
        }
        .student-reg-card-sub {
            margin: 0;
            color: #95A0B2;
            font-size: 0.92rem;
        }
        .student-reg-step-shell {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.85rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 0.95rem;
        }
        .student-reg-step {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-width: 0;
        }
        .student-reg-step-circle {
            width: 32px;
            height: 32px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.92rem;
            font-weight: 700;
            background: rgba(255,255,255,0.07);
            color: #B5BDC9;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .student-reg-step.active .student-reg-step-circle {
            background: linear-gradient(180deg, #2B92FF 0%, #2C69E3 100%);
            color: white;
            border-color: rgba(90, 163, 255, 0.6);
        }
        .student-reg-step-label {
            color: #AEB6C3;
            font-size: 0.95rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .student-reg-step.active .student-reg-step-label {
            color: #F4F7FB;
        }
        .student-reg-step-line {
            flex: 1 1 auto;
            height: 2px;
            min-width: 28px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
        }
        .student-reg-info {
            border-radius: 14px;
            border: 1px solid rgba(49,140,255,0.16);
            background: linear-gradient(180deg, rgba(22,66,132,0.34) 0%, rgba(18,42,86,0.28) 100%);
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            color: #75B4FF;
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .student-reg-side-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .student-reg-mini {
            border-radius: 14px;
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.9rem 0.95rem;
        }
        .student-reg-mini-label {
            color: #9AA5B8;
            font-size: 0.88rem;
            margin-bottom: 0.25rem;
        }
        .student-reg-mini-value {
            color: #F5F7FB;
            font-size: 1.02rem;
            font-weight: 700;
        }
        .student-reg-progress-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            overflow: hidden;
            margin-top: 0.7rem;
        }
        .student-reg-progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2D93FF 0%, #58B7FF 100%);
        }
        .student-reg-pose-list {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.55rem;
            margin: 0.9rem 0 1rem 0;
        }
        .student-reg-pose-pill {
            border-radius: 14px;
            padding: 0.72rem 0.45rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            background: rgba(255,255,255,0.025);
        }
        .student-reg-pose-pill.active {
            border-color: rgba(70,151,255,0.35);
            background: linear-gradient(180deg, rgba(34,80,155,0.26) 0%, rgba(22,45,88,0.18) 100%);
        }
        .student-reg-pose-num {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            margin: 0 auto 0.35rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.84rem;
            font-weight: 700;
            background: rgba(255,255,255,0.08);
            color: #B8C0CC;
        }
        .student-reg-pose-pill.active .student-reg-pose-num {
            background: linear-gradient(180deg, #2B92FF 0%, #2C69E3 100%);
            color: white;
        }
        .student-reg-pose-label {
            color: #B3BCC9;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .student-reg-pose-pill.active .student-reg-pose-label {
            color: #F4F7FB;
        }
        .student-reg-lock {
            color: #96A0AF;
            font-size: 0.92rem;
            margin-top: 0.8rem;
        }
        .student-reg-camera-wrap {
            margin-top: 1rem;
        }
        .student-reg-camera-placeholder {
            min-height: 360px;
            border: 1px dashed rgba(109, 101, 255, 0.34);
            border-radius: 18px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #A6AFBD;
            text-align: center;
            padding: 1.2rem;
            background: linear-gradient(180deg, rgba(62,43,118,0.13) 0%, rgba(23,27,38,0.18) 100%);
        }
        .student-reg-camera-icon {
            width: 62px;
            height: 62px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, rgba(102,83,255,0.4) 0%, rgba(88,67,219,0.3) 100%);
            margin-bottom: 0.85rem;
            font-size: 1.6rem;
            color: #ECEBFF;
        }
        @media (max-width: 1100px) {
            .student-reg-pose-list {
                grid-template-columns: repeat(2, 1fr);
            }
            .student-reg-side-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_student_registration_header() -> None:
    st.markdown(
        f"""
        <div class="student-reg-header">
            <div class="student-reg-icon">{_students_group_icon_svg()}</div>
            <div>
                <h1 class="student-reg-title">Cadastro de Alunos</h1>
                <p class="student-reg-subtitle">Preencha os dados do aluno e capture as poses para habilitar o monitoramento.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_student_registration_card_header(icon: str, title: str, subtitle: str, gradient: str) -> None:
    st.markdown(
        (
            "<div class='student-reg-card-title'>"
            f"<div class='student-reg-card-icon' style='background:{gradient};'>{html.escape(icon)}</div>"
            "<div>"
            f"<p class='student-reg-card-heading'>{html.escape(title)}</p>"
            f"<p class='student-reg-card-sub'>{html.escape(subtitle)}</p>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_student_registration_stepper(active_step: int) -> None:
    first_class = "student-reg-step active" if active_step == 1 else "student-reg-step"
    second_class = "student-reg-step active" if active_step == 2 else "student-reg-step"
    st.markdown(
        (
            "<div class='student-reg-step-shell'>"
            f"<div class='{first_class}'>"
            "<div class='student-reg-step-circle'>1</div>"
            "<div class='student-reg-step-label'>Identificação</div>"
            "</div>"
            "<div class='student-reg-step-line'></div>"
            f"<div class='{second_class}'>"
            "<div class='student-reg-step-circle'>2</div>"
            "<div class='student-reg-step-label'>Captura de Poses</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_capture_pose_list(pose_index: int, poses: list[str]) -> None:
    items = []
    for idx, pose in enumerate(poses, start=1):
        css_class = "student-reg-pose-pill active" if idx - 1 == pose_index else "student-reg-pose-pill"
        items.append(
            f"<div class='{css_class}'>"
            f"<div class='student-reg-pose-num'>{idx}</div>"
            f"<div class='student-reg-pose-label'>{html.escape(CAPTURE_POSE_LABELS.get(pose, pose.title()))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='student-reg-pose-list'>{''.join(items)}</div>", unsafe_allow_html=True)


def _inject_sidebar_menu_styles() -> None:
    monitor_icon_data_uri = _svg_to_data_uri(_monitor_hero_icon_svg())
    students_icon_data_uri = _svg_to_data_uri(_students_group_icon_svg())
    charts_icon_data_uri = _svg_to_data_uri(_charts_menu_icon_svg())
    reports_icon_data_uri = _svg_to_data_uri(_reports_menu_icon_svg())
    css = """
        <style>
        .stApp [data-testid="stMainBlockContainer"] {
            padding-top: 1rem;
            padding-bottom: 1.4rem;
        }
        .stApp header[data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] > div:first-child {
            background:
                radial-gradient(circle at top left, rgba(78, 92, 140, 0.20), transparent 28%),
                linear-gradient(180deg, #141824 0%, #10141E 100%);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.04);
        }
        .sidebar-shell {
            padding: 0.25rem 0.15rem 0.4rem 0.15rem;
        }
        .sidebar-hero-img {
            width: 100%;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 40px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
            display: block;
        }
        .sidebar-profile {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 0.3rem 0 0.85rem 0;
        }
        .sidebar-avatar {
            width: 62px;
            height: 62px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.65rem;
            background: linear-gradient(180deg, #6C7BFF 0%, #5A48E1 100%);
            box-shadow: inset 0 8px 18px rgba(255,255,255,0.16), 0 10px 28px rgba(72, 54, 176, 0.28);
        }
        .sidebar-profile-name {
            margin: 0;
            font-size: 1.15rem;
            line-height: 1.1;
            color: #F4F6FB;
            font-weight: 800;
        }
        .sidebar-profile-role {
            margin: 0.28rem 0 0 0;
            color: #A3ABBB;
            font-size: 0.95rem;
        }
        .sidebar-profile-divider {
            height: 1px;
            background: rgba(255,255,255,0.08);
            margin: 0.25rem 0 0.95rem 0;
        }
        .sidebar-menu-label {
            margin: 0.2rem 0 0.55rem 0;
            color: #8E97A9;
            font-size: 0.8rem;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            font-weight: 700;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            width: 100%;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            justify-content: center;
            position: relative;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.06);
            background: linear-gradient(180deg, rgba(28,33,46,0.92) 0%, rgba(22,27,39,0.96) 100%);
            padding: 0.95rem 4rem 0.95rem 1.1rem;
            margin: 0 0 0.8rem 0;
            transition: all 0.2s ease;
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
            width: 100%;
            min-width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            min-height: 112px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: rgba(117, 103, 255, 0.24);
            transform: translateY(-1px);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            border-color: rgba(115, 100, 255, 0.34);
            background: linear-gradient(180deg, rgba(33,36,66,0.96) 0%, rgba(26,28,51,0.96) 100%);
            box-shadow: 0 14px 30px rgba(53, 45, 110, 0.22);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::before {
            content: "";
            position: absolute;
            left: 0;
            top: 14px;
            bottom: 14px;
            width: 4px;
            border-radius: 999px;
            background: linear-gradient(180deg, #7E6BFF 0%, #6553E8 100%);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label::after {
            content: "›";
            position: absolute;
            right: 1.2rem;
            top: 50%;
            transform: translateY(-50%);
            color: #8F98AA;
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 500;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::after {
            color: #7E6BFF;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
            display: none;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child {
            width: 100%;
            display: block;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label p {
            white-space: pre-line;
            margin: 0;
            line-height: 1.32;
            color: #BCC4D2;
            font-weight: 600;
            font-size: 1rem;
            position: relative;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) p {
            padding-left: 2.2rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) p::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.02rem;
            width: 1.6rem;
            height: 1.6rem;
            background-image: url("__STUDENTS_ICON_DATA_URI__");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) p {
            padding-left: 2.2rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) p::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.02rem;
            width: 1.6rem;
            height: 1.6rem;
            background-image: url("__MONITOR_ICON_DATA_URI__");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) p {
            padding-left: 2.2rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) p::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.02rem;
            width: 1.6rem;
            height: 1.6rem;
            background-image: url("__CHARTS_ICON_DATA_URI__");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) p {
            padding-left: 2.2rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) p::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0.02rem;
            width: 1.6rem;
            height: 1.6rem;
            background-image: url("__REPORTS_ICON_DATA_URI__");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label p::first-line {
            color: #F3F6FA;
            font-size: 1.18rem;
            font-weight: 800;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label p {
            letter-spacing: 0.005em;
        }
        .sidebar-help-box {
            margin-top: 1rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.07);
            background: linear-gradient(180deg, rgba(25,30,43,0.88) 0%, rgba(20,24,35,0.94) 100%);
            padding: 0.95rem 1rem;
            width: 100%;
            min-width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            display: block;
        }
        .sidebar-help-title {
            color: #D5DAE3;
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sidebar-help-subtitle {
            color: #929CAF;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 62px;
            border-radius: 20px;
            border: 1px solid rgba(132,148,255,0.26);
            background: linear-gradient(180deg, rgba(43,49,76,0.98) 0%, rgba(28,33,52,0.98) 100%);
            color: #FFFFFF;
            font-size: 1.7rem;
            font-weight: 800;
            text-shadow: 0 0 10px rgba(255,255,255,0.18);
            box-shadow: 0 12px 28px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(161,176,255,0.42);
            color: #FFFFFF;
            transform: translateY(-1px);
        }
        .sidebar-help-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        .sidebar-help-icon {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid rgba(116, 86, 255, 0.8);
            color: #8B70FF;
            font-size: 1.15rem;
            font-weight: 800;
            flex: 0 0 auto;
        }
        </style>
    """
    st.markdown(
        css.replace("__MONITOR_ICON_DATA_URI__", monitor_icon_data_uri)
        .replace("__STUDENTS_ICON_DATA_URI__", students_icon_data_uri)
        .replace("__CHARTS_ICON_DATA_URI__", charts_icon_data_uri)
        .replace("__REPORTS_ICON_DATA_URI__", reports_icon_data_uri),
        unsafe_allow_html=True,
    )


def img_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_runtime_diagnostics():
    diagnostics = {
        "python": sys.executable,
        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", "(nao definido)"),
        "nvidia_driver_capabilities": os.getenv("NVIDIA_DRIVER_CAPABILITIES", "(nao definido)"),
        "torch_version": "(indisponivel)",
        "torch_cuda_version": "(indisponivel)",
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "torch_device_name": "(nenhuma GPU visivel)",
        "onnxruntime_version": "(indisponivel)",
        "onnxruntime_providers": [],
    }

    try:
        diagnostics["torch_version"] = torch.__version__
        diagnostics["torch_cuda_version"] = torch.version.cuda or "(sem CUDA no build)"
        diagnostics["torch_cuda_available"] = torch.cuda.is_available()
        diagnostics["torch_device_count"] = torch.cuda.device_count()
        if diagnostics["torch_cuda_available"] and diagnostics["torch_device_count"] > 0:
            diagnostics["torch_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        diagnostics["torch_device_name"] = f"erro: {exc}"

    try:
        import onnxruntime as ort

        diagnostics["onnxruntime_version"] = ort.__version__
        diagnostics["onnxruntime_providers"] = ort.get_available_providers()
    except Exception as exc:
        diagnostics["onnxruntime_providers"] = [f"erro: {exc}"]

    return diagnostics


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = repr(exc).lower()
    return "cuda" in message and (
        "busy" in message
        or "unavailable" in message
        or "device-side assert" in message
        or "no kernel image" in message
        or "invalid device" in message
        or "acceleratorerror" in message
    )


def _create_pose_model(device: str):
    pose_model_name = "yolo11m-pose.pt" if device == "cuda" else "yolo11n-pose.pt"
    return YOLO(pose_model_name)


def _create_face_model(device: str):
    if device == "cuda":
        model_face = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model_face.prepare(ctx_id=0, det_size=FACE_DET_SIZE_GPU)
        return model_face

    model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model_face.prepare(ctx_id=-1, det_size=FACE_DET_SIZE_CPU)
    return model_face


@st.cache_resource(show_spinner=False)
def build_monitor_runtime(device: str, relay_host: str, relay_port: int):
    runtime_device = device
    try:
        model = _create_pose_model(runtime_device)
        model_face = _create_face_model(runtime_device)
    except Exception as exc:
        if runtime_device != "cuda" or not _is_cuda_runtime_error(exc):
            raise
        runtime_device = "cpu"
        model = _create_pose_model(runtime_device)
        model_face = _create_face_model(runtime_device)

    known_face_encodings, known_face_names = load_insightface_data()
    known_face_encodings_norm = (
        known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
    ) if len(known_face_encodings) > 0 else None

    video_stream = VideoStream((relay_host, relay_port)).start()
    pose_imgsz = POSE_IMGSZ_GPU if runtime_device == "cuda" else POSE_IMGSZ_CPU
    detector = DetectorWorker(
        model,
        model_face,
        runtime_device,
        min_inference_interval=0.08 if runtime_device == "cuda" else 0.18,
        face_refresh_interval=FACE_REFRESH_INTERVAL_GPU if runtime_device == "cuda" else FACE_REFRESH_INTERVAL_CPU,
        pose_imgsz=pose_imgsz,
        pose_conf=POSE_DET_CONF,
    ).start()

    return {
        "model": model,
        "model_face": model_face,
        "known_face_encodings_norm": known_face_encodings_norm,
        "known_face_names": known_face_names,
        "video_stream": video_stream,
        "detector": detector,
        "runtime_device": runtime_device,
    }


def teardown_monitor_runtime():
    runtime = st.session_state.pop("monitor_runtime", None)
    if isinstance(runtime, dict):
        detector = runtime.get("detector")
        video_stream = runtime.get("video_stream")
        if detector is not None:
            try:
                detector.stop()
            except Exception:
                pass
        if video_stream is not None:
            try:
                video_stream.stop()
            except Exception:
                pass
    build_monitor_runtime.clear()


def _friendly_session_status(status: str | None) -> str:
    normalized = (status or "").strip().lower()
    if normalized == SESSION_STATUS_OPEN:
        return "Em andamento"
    if normalized == SESSION_STATUS_CLOSED:
        return "Encerrado"
    return "Não iniciado"


def _status_badge_markup(status: str | None) -> str:
    palette = {
        "Não iniciado": ("#9AA0AA", "rgba(154,160,170,0.14)", "rgba(154,160,170,0.35)"),
        "Em andamento": ("#3DDC97", "rgba(61,220,151,0.14)", "rgba(61,220,151,0.35)"),
        "Encerrado": ("#E57373", "rgba(229,115,115,0.14)", "rgba(229,115,115,0.35)"),
    }
    normalized = status if status in palette else _friendly_session_status(status)
    fg, bg, border = palette[normalized]
    return (
        f"<span style='display:inline-flex; align-items:center; padding:0.22rem 0.7rem; "
        f"border-radius:999px; border:1px solid {border}; background:{bg}; color:{fg}; "
        f"font-size:0.85rem; font-weight:600;'>{normalized}</span>"
    )


def _format_datetime_br(value) -> str:
    if not value:
        return "-"
    return pd.to_datetime(value).strftime("%H:%M:%S")


def _format_duration_label(delta: datetime.timedelta | None) -> str:
    if delta is None:
        return "-"
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _security_camera_icon_svg() -> str:
    return """
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M14 23.5C14 21.6 15.6 20 17.5 20H38.8C40 20 41.2 19.7 42.2 19.1L49.8 14.6C51.7 13.5 54 14.8 54 17V31.4C54 33.6 51.7 34.9 49.8 33.8L42.2 29.3C41.2 28.7 40 28.4 38.8 28.4H17.5C15.6 28.4 14 26.9 14 25V23.5Z" fill="#EEF3FF" fill-opacity="0.96"/>
        <path d="M21 31.5H28L25.6 37.8H18.4L21 31.5Z" fill="#C8D4EA"/>
        <path d="M25.2 37.8H36.8C38.2 37.8 39.4 39 39.4 40.4C39.4 41.8 38.2 43 36.8 43H18.2C16.8 43 15.6 41.8 15.6 40.4C15.6 39 16.8 37.8 18.2 37.8H25.2Z" fill="#D8E1F2"/>
        <circle cx="34.5" cy="24.2" r="5.8" fill="#6E58FF"/>
        <circle cx="34.5" cy="24.2" r="3.5" fill="#1C2340"/>
        <circle cx="34.5" cy="24.2" r="1.6" fill="#9CD6FF"/>
        <path d="M46.8 21L51.6 18.2" stroke="#B9C6DD" stroke-width="2.6" stroke-linecap="round"/>
        <path d="M44.8 25.2H50.8" stroke="#B9C6DD" stroke-width="2.6" stroke-linecap="round"/>
        <path d="M46.8 29.4L51.6 32.2" stroke="#B9C6DD" stroke-width="2.6" stroke-linecap="round"/>
    </svg>
    """


def _students_group_icon_svg() -> str:
    return """
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M9 17L24 13L39 17L24 21L9 17Z" stroke="#F4F7FF" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="M17 21V30.5C17 31.8 16.4 33 15.4 33.9L14.9 34.4C14.1 35.1 13.7 36.1 13.7 37.1V40.5C13.7 46.8 18.7 51.8 25 51.8C31.3 51.8 36.3 46.8 36.3 40.5V37.1C36.3 36.1 35.9 35.1 35.1 34.4L34.6 33.9C33.6 33 33 31.8 33 30.5V21" stroke="#F4F7FF" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M17 24.5C19.4 26.1 22 26.9 24.8 26.9C27.7 26.9 30.5 26.1 33 24.4" stroke="#F4F7FF" stroke-width="3.2" stroke-linecap="round"/>
        <path d="M12 62V59.6C12 54.2 16.4 49.8 21.8 49.8H28.2C33.6 49.8 38 54.2 38 59.6V62" stroke="#F4F7FF" stroke-width="3.2" stroke-linecap="round"/>
        <path d="M31.5 17L46.5 13.8L58.5 17L43.5 20.2L31.5 17Z" stroke="#A9D7FF" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="M41 20.2V29.2C41 30.3 41.4 31.3 42.1 32.1L42.9 33C43.5 33.7 43.9 34.6 44 35.5C44.3 39.9 45.6 43 47.8 45.1C45.8 47 43.3 48.1 40.6 48.1C34.4 48.1 29.4 43.1 29.4 36.9V33.9C29.4 33 29.7 32.2 30.3 31.6L31 30.8C31.7 30 32.1 29 32.1 27.9V20.4" stroke="#A9D7FF" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M32.6 23.8C34.9 25.1 37.1 25.8 39.4 25.8C42 25.8 44.5 25 47 23.5" stroke="#A9D7FF" stroke-width="3.2" stroke-linecap="round"/>
        <path d="M51.5 18.6V30.6" stroke="#A9D7FF" stroke-width="3.2" stroke-linecap="round"/>
        <path d="M51.5 30.6L49.8 35H53.2L51.5 30.6Z" stroke="#A9D7FF" stroke-width="3" stroke-linejoin="round"/>
        <path d="M33.5 62V59.7C33.5 55.2 37.2 51.5 41.7 51.5H47.1C51.6 51.5 55.3 55.2 55.3 59.7V62" stroke="#A9D7FF" stroke-width="3.2" stroke-linecap="round"/>
    </svg>
    """


def _charts_menu_icon_svg() -> str:
    return """
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <rect x="12" y="10" width="40" height="44" rx="8" fill="#EEF3FF" fill-opacity="0.96"/>
        <path d="M21 22V45" stroke="#28324D" stroke-width="3" stroke-linecap="round"/>
        <path d="M21 45H45" stroke="#28324D" stroke-width="3" stroke-linecap="round"/>
        <rect x="24.5" y="31" width="5.8" height="14" rx="1.5" fill="#52A7FF"/>
        <rect x="33" y="25" width="5.8" height="20" rx="1.5" fill="#7DDB78"/>
        <rect x="41.5" y="18" width="5.8" height="27" rx="1.5" fill="#F0B54D"/>
        <path d="M18 15.5H46" stroke="#BCC9E0" stroke-width="2.4" stroke-linecap="round"/>
    </svg>
    """


def _reports_menu_icon_svg() -> str:
    return """
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M18 10H39L48 19V50C48 52.2 46.2 54 44 54H18C15.8 54 14 52.2 14 50V14C14 11.8 15.8 10 18 10Z" fill="#EEF3FF" fill-opacity="0.96"/>
        <path d="M39 10V18C39 19.7 40.3 21 42 21H48" fill="#D9E4F7"/>
        <path d="M39 10V18C39 19.7 40.3 21 42 21H48" stroke="#B6C5DE" stroke-width="2.2" stroke-linejoin="round"/>
        <path d="M22 24H39" stroke="#2C3653" stroke-width="2.6" stroke-linecap="round"/>
        <path d="M22 31H39" stroke="#2C3653" stroke-width="2.6" stroke-linecap="round"/>
        <path d="M22 38H34" stroke="#2C3653" stroke-width="2.6" stroke-linecap="round"/>
        <circle cx="43.5" cy="28.5" r="2.2" fill="#52A7FF"/>
        <circle cx="43.5" cy="35.5" r="2.2" fill="#7DDB78"/>
        <circle cx="43.5" cy="42.5" r="2.2" fill="#F0B54D"/>
    </svg>
    """


def _monitor_hero_icon_svg() -> str:
    return """
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M33 10C24.2 10 17 17.2 17 26V30.2C17 33.4 15.8 36.5 13.7 38.9L11.8 41.1C10.3 42.8 11.5 45.5 13.8 45.5H52.2C54.5 45.5 55.7 42.8 54.2 41.1L52.3 38.9C50.2 36.5 49 33.4 49 30.2V26C49 17.2 41.8 10 33 10Z" fill="#E9EEF9" fill-opacity="0.96"/>
        <path d="M24.5 45.5C25.5 50.2 29 53 33 53C37 53 40.5 50.2 41.5 45.5H24.5Z" fill="#DCE4F4"/>
        <path d="M23 22.5C25.8 18.2 30.2 15.8 35.2 15.8C37.4 15.8 39.6 16.3 41.5 17.2" stroke="#B9C5DA" stroke-width="3" stroke-linecap="round"/>
        <path d="M16.2 21.8C18 18.7 20.7 16.1 23.9 14.4" stroke="#8AAAF6" stroke-width="3.2" stroke-linecap="round"/>
        <circle cx="45.5" cy="17.5" r="2.8" fill="#6E58FF"/>
        <path d="M29 48.6C30.1 50.3 31.4 51.1 33 51.1C34.6 51.1 35.9 50.3 37 48.6" stroke="#B7C3D9" stroke-width="2.6" stroke-linecap="round"/>
    </svg>
    """


def _svg_to_data_uri(svg: str) -> str:
    return f"data:image/svg+xml;utf8,{quote(svg.strip())}"


def _render_context_card(session_state_label: str, selected_subject_label: str, selected_class_label: str, session_data=None):
    start_time = None if not session_data else pd.to_datetime(session_data.get("start_time")) if session_data.get("start_time") else None
    end_time = None if not session_data else pd.to_datetime(session_data.get("end_time")) if session_data.get("end_time") else None

    elapsed_delta = None
    total_delta = None
    if start_time is not None:
        if session_state_label == "Em andamento":
            elapsed_delta = pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None).to_pydatetime() - start_time.to_pydatetime()
        elif end_time is not None:
            total_delta = end_time.to_pydatetime() - start_time.to_pydatetime()

    session_value = f"ID {session_data['id']}" if session_data and session_state_label == "Em andamento" else "-"
    start_value = _format_datetime_br(start_time)
    secondary_label = "Início"
    secondary_value = start_value
    if session_state_label == "Encerrado":
        secondary_label = "Duração total"
        secondary_value = _format_duration_label(total_delta)

    st.markdown(
        f"""
        <div class="monitor-context-card">
            <div class="monitor-card-head">
                <div class="monitor-card-title-wrap">
                    <div class="monitor-card-icon monitor-card-icon-neutral">▣</div>
                    <div class="monitor-card-title">Contexto da Aula</div>
                </div>
                {_status_badge_markup(session_state_label)}
            </div>
            <div class="monitor-context-grid">
                <div class="monitor-context-item">
                    <div class="monitor-context-label">Disciplina</div>
                    <div class="monitor-context-value">{html.escape(selected_subject_label or "-")}</div>
                </div>
                <div class="monitor-context-item">
                    <div class="monitor-context-label">Turma</div>
                    <div class="monitor-context-value">{html.escape(selected_class_label or "-")}</div>
                </div>
                <div class="monitor-context-item">
                    <div class="monitor-context-label">Sessão ativa</div>
                    <div class="monitor-context-value">{html.escape(session_value)}</div>
                </div>
                <div class="monitor-context-item">
                    <div class="monitor-context-label">{secondary_label}</div>
                    <div class="monitor-context-value">{html.escape(secondary_value)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_monitor_video_snapshot(current_session=None) -> dict:
    runtime = st.session_state.get("monitor_runtime")
    video_stream = None if runtime is None else runtime.get("video_stream")
    status = video_stream.get_status() if video_stream is not None and hasattr(video_stream, "get_status") else {}
    frames_received = int(status.get("frames_received", 0) or 0)
    live_stats = st.session_state.get("monitor_live_stats", {})

    if current_session and current_session.get("status") == SESSION_STATUS_OPEN and current_session.get("start_time"):
        start_time = pd.to_datetime(current_session["start_time"]).to_pydatetime()
        duration = pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None).to_pydatetime() - start_time
        state_label = "Em andamento"
        badge_text = "Sessão ativa"
        badge_class = "live"
    elif current_session and current_session.get("start_time") and current_session.get("end_time"):
        start_time = pd.to_datetime(current_session["start_time"]).to_pydatetime()
        end_time = pd.to_datetime(current_session["end_time"]).to_pydatetime()
        duration = end_time - start_time
        state_label = "Sessão encerrada"
        badge_text = "Sessão encerrada"
        badge_class = "ended"
    else:
        duration = datetime.timedelta(0)
        state_label = "Aguardando início"
        badge_text = "Aguardando início"
        badge_class = "waiting"

    return {
        "frames_received": frames_received,
        "recognized_now": int(live_stats.get("recognized_faces_count", 0) or 0),
        "detected_now": int(live_stats.get("detected_faces_count", 0) or 0),
        "duration_label": _format_duration_label(duration),
        "state_label": state_label,
        "badge_text": badge_text,
        "badge_class": badge_class,
    }


def _render_monitor_video_summary_markup(video_snapshot: dict) -> None:
    return f"""
    <div class="monitor-video-head" style="padding:0 0 1rem 0; border-bottom:none;">
        <div class="monitor-card-title-wrap">
            <div class="monitor-card-icon monitor-card-icon-purple">{_monitor_hero_icon_svg()}</div>
            <div class="monitor-card-title">Vídeo de Monitoramento</div>
        </div>
        <div class="monitor-video-badge {video_snapshot['badge_class']}">
            <span>●</span>
            <span>{html.escape(video_snapshot['badge_text'])}</span>
        </div>
    </div>
    <div class="monitor-video-metrics" style="padding:0 0 1rem 0;">
        <div class="monitor-metric-box">
            <div class="monitor-metric-label">Status</div>
            <div class="monitor-metric-value">{html.escape(video_snapshot['state_label'])}</div>
        </div>
        <div class="monitor-metric-box">
            <div class="monitor-metric-label">Duração da sessão</div>
            <div class="monitor-metric-value">{html.escape(video_snapshot['duration_label'])}</div>
        </div>
        <div class="monitor-metric-box">
            <div class="monitor-metric-label">Alunos reconhecidos agora</div>
            <div class="monitor-metric-value">{video_snapshot['recognized_now']}</div>
        </div>
    </div>
    """


def _update_recognized_students_metric(recognized_students_now: set[str]) -> int:
    now = time.time()
    remembered = st.session_state.get("monitor_recognized_students", {})
    if not isinstance(remembered, dict):
        remembered = {}

    refreshed = {
        name: ts
        for name, ts in remembered.items()
        if (now - float(ts)) <= MONITOR_RECOGNIZED_HOLD_SECONDS
    }
    for name in recognized_students_now:
        refreshed[name] = now

    st.session_state["monitor_recognized_students"] = refreshed
    return len(refreshed)


def render_monitor_video_summary_fragment(current_session_id=None, placeholder=None):
    current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None
    video_snapshot = _get_monitor_video_snapshot(current_session)
    markup = _render_monitor_video_summary_markup(video_snapshot)
    if placeholder is None:
        st.markdown(markup, unsafe_allow_html=True)
        return

    cached = st.session_state.get("monitor_summary_html")
    if cached == markup:
        return
    st.session_state["monitor_summary_html"] = markup
    placeholder.markdown(markup, unsafe_allow_html=True)


def render_context_fragment(session_state_label: str, selected_subject_label: str, selected_class_label: str, session_data=None):
    _render_context_card(session_state_label, selected_subject_label, selected_class_label, session_data)


def _render_monitor_frame(
    frame_placeholder,
    frame_bgr,
    frame_id: int,
    jpeg_quality: int = 52,
    jpeg_bytes: bytes | None = None,
    overlays: list[dict] | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> None:
    if frame_placeholder is None:
        return
    cached = st.session_state.get("monitor_last_display_frame")
    if (
        isinstance(cached, dict)
        and cached.get("frame_id") == frame_id
        and cached.get("html")
    ):
        return

    if frame_bgr is not None:
        frame_h, frame_w = frame_bgr.shape[:2]
        display_frame = frame_bgr
        max_display_width = 1280
        if frame_w > max_display_width:
            scale = max_display_width / float(frame_w)
            resized_h = max(1, int(frame_h * scale))
            display_frame = cv2.resize(frame_bgr, (max_display_width, resized_h), interpolation=cv2.INTER_AREA)
    elif frame_shape is not None:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        display_frame = None
        max_display_width = 1280
    else:
        return

    if jpeg_bytes is None or (frame_bgr is not None and frame_w > max_display_width):
        ok, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            return
        jpeg_bytes = jpg.tobytes()

    jpg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    overlay_html = ""
    for overlay in overlays or []:
        x1 = max(0.0, min(100.0, (float(overlay["x1"]) / max(1, frame_w)) * 100.0))
        y1 = max(0.0, min(100.0, (float(overlay["y1"]) / max(1, frame_h)) * 100.0))
        x2 = max(0.0, min(100.0, (float(overlay["x2"]) / max(1, frame_w)) * 100.0))
        y2 = max(0.0, min(100.0, (float(overlay["y2"]) / max(1, frame_h)) * 100.0))
        width = max(0.5, x2 - x1)
        height = max(0.5, y2 - y1)
        color = html.escape(str(overlay.get("color", "#22c55e")))
        label_bg = html.escape(str(overlay.get("label_bg", color)))
        label_fg = html.escape(str(overlay.get("label_fg", "#ffffff")))
        label = html.escape(str(overlay.get("label", "")))
        label_top = max(0.0, y1 - 5.0)
        overlay_html += (
            f"<div style='position:absolute; left:{x1:.3f}%; top:{y1:.3f}%; "
            f"width:{width:.3f}%; height:{height:.3f}%; border:2px solid {color}; "
            "border-radius:10px; box-sizing:border-box; pointer-events:none;'></div>"
        )
        if label:
            overlay_html += (
                f"<div style='position:absolute; left:{x1:.3f}%; top:{label_top:.3f}%; "
                f"background:{label_bg}; color:{label_fg}; padding:4px 8px; border-radius:8px; "
                "font-size:12px; font-weight:600; line-height:1.1; white-space:nowrap; "
                "box-sizing:border-box; pointer-events:none;'>"
                f"{label}</div>"
            )
    frame_html = (
        "<div style='width:100%; min-height:430px; display:flex; align-items:center; justify-content:center;'>"
        "<div style='position:relative; width:100%;'>"
        f"<img src='data:image/jpeg;base64,{jpg_b64}' "
        "style='display:block; width:100%; height:auto; border-radius:18px;' "
        "alt='Monitoramento em tempo real' />"
        f"{overlay_html}"
        "</div>"
        "</div>"
    )
    st.session_state["monitor_last_display_frame"] = {
        "frame_id": frame_id,
        "html": frame_html,
    }
    frame_placeholder.markdown(frame_html, unsafe_allow_html=True)


def _render_monitor_stream_placeholder(frame_placeholder, title: str, subtitle: str) -> None:
    if frame_placeholder is None:
        return
    placeholder_html = f"""
    <div class="monitor-placeholder">
        <div>
            <div class="monitor-placeholder-icon">{_monitor_hero_icon_svg()}</div>
            <div class="monitor-placeholder-title">{html.escape(title)}</div>
            <div class="monitor-placeholder-subtitle">{html.escape(subtitle)}</div>
        </div>
    </div>
    """
    cached = st.session_state.get("monitor_last_display_frame")
    if isinstance(cached, dict) and cached.get("html") == placeholder_html:
        return
    st.session_state["monitor_last_display_frame"] = {
        "frame_id": "placeholder",
        "html": placeholder_html,
    }
    frame_placeholder.markdown(placeholder_html, unsafe_allow_html=True)


def _update_monitor_status(status_placeholder, level: str, message: str | None = None) -> None:
    previous = st.session_state.get("monitor_status_message")
    if level == "empty":
        if previous is None:
            return
        status_placeholder.empty()
        st.session_state.pop("monitor_status_message", None)
        return

    payload = {"level": level, "message": message or ""}
    if previous == payload:
        return

    if level == "warning":
        status_placeholder.warning(payload["message"])
    elif level == "error":
        status_placeholder.error(payload["message"])
    else:
        status_placeholder.info(payload["message"])
    st.session_state["monitor_status_message"] = payload


def render_monitor_preview_fragment(frame_placeholder):
    runtime = st.session_state.get("monitor_runtime")
    video_stream = None if runtime is None else runtime.get("video_stream")
    if video_stream is None or frame_placeholder is None:
        return
    if not hasattr(video_stream, "read_jpeg_with_meta"):
        return
    frame_jpeg, frame_id, last_frame_at = video_stream.read_jpeg_with_meta()
    if last_frame_at is not None and (time.time() - float(last_frame_at)) > MONITOR_FRAME_STALE_SECONDS:
        _render_monitor_stream_placeholder(
            frame_placeholder,
            "Sinal de video interrompido",
            "O relay continua ativo, mas nao recebeu frame novo do RTSP recentemente.",
        )
        return
    if frame_jpeg is None:
        return
    overlays = st.session_state.get("monitor_last_overlays", [])
    frame_shape = st.session_state.get("monitor_last_frame_shape")
    _render_monitor_frame(
        frame_placeholder,
        frame_bgr=None,
        frame_id=frame_id,
        jpeg_bytes=frame_jpeg,
        overlays=overlays,
        frame_shape=frame_shape,
    )


@st.fragment(run_every=MONITOR_UI_REFRESH_SECONDS)
def process_monitor_fragment(
    school: str,
    discipline: str,
    user_name: str,
    current_session_id,
    confidence_threshold: float,
    show_debug: bool,
    debug_font: float,
    box_margin_ratio: float,
    show_unknown_boxes: bool,
    summary_placeholder,
    status_placeholder,
    frame_placeholder,
):
    runtime = st.session_state.get("monitor_runtime")
    video_stream = None if runtime is None else runtime.get("video_stream")
    detector = None if runtime is None else runtime.get("detector")
    episode_manager = st.session_state.get("episode_manager")
    known_face_encodings_norm = None if runtime is None else runtime.get("known_face_encodings_norm")
    known_face_names = [] if runtime is None else runtime.get("known_face_names", [])
    student_lookup = st.session_state.get("student_lookup", {})

    frame = None
    frame_jpeg = None
    frame_id = 0
    frame_last_at = None
    st.session_state["monitor_live_stats"] = {
        "detected_faces_count": 0,
        "recognized_faces_count": 0,
        "rendered_tracks_count": 0,
    }
    if video_stream is not None:
        if hasattr(video_stream, "read_latest_with_meta"):
            frame, frame_jpeg, frame_id, frame_last_at = video_stream.read_latest_with_meta()
        elif hasattr(video_stream, "read_with_meta"):
            frame, frame_id, frame_last_at = video_stream.read_with_meta()
            if hasattr(video_stream, "read_jpeg_with_meta"):
                frame_jpeg, _, frame_last_at = video_stream.read_jpeg_with_meta()
        else:
            frame = video_stream.read()
            status = video_stream.get_status() if hasattr(video_stream, "get_status") else {}
            frame_id = status.get("frame_id", status.get("frames_received", 0))
            frame_last_at = status.get("last_frame_at")

    frame_stale = (
        frame_last_at is not None
        and (time.time() - float(frame_last_at)) > MONITOR_FRAME_STALE_SECONDS
    )
    if frame_stale:
        frame = None
        frame_jpeg = None

    if frame is None:
        status = video_stream.get_status() if video_stream is not None else {
            "server": (RELAY_HOST, RELAY_PORT),
            "connected": False,
            "frames_received": 0,
            "last_frame_at": None,
            "last_error": "video_stream ausente",
        }
        waited = time.time() - st.session_state.get("monitor_waiting_since", time.time())
        last_frame_age = None
        if status["last_frame_at"] is not None:
            last_frame_age = time.time() - status["last_frame_at"]

        _update_monitor_status(
            status_placeholder,
            "warning",
            "\n".join(
                [
                    f"Aguardando frames do relay `{status['server'][0]}:{status['server'][1]}`",
                    f"Conectado: {'sim' if status['connected'] else 'nao'}",
                    f"Frames recebidos: {status['frames_received']}",
                    f"Ultimo frame ha: {f'{last_frame_age:.1f}s' if last_frame_age is not None else 'nenhum'}",
                    f"Ultimo erro: {status['last_error'] or 'nenhum'}",
                ]
            ),
        )
        if frame_stale and frame_placeholder is not None:
            _render_monitor_stream_placeholder(
                frame_placeholder,
                "Sinal de video interrompido",
                "O ultimo frame recebido ficou antigo. Verifique o RTSP e os logs do relay.",
            )
        if waited > 10:
            _update_monitor_status(
                status_placeholder,
                "error",
                "O app conectou no relay, mas nao recebeu frame util a tempo. "
                "Valide os logs do container `relay` e a estabilidade do RTSP.",
            )
        return False

    st.session_state["monitor_waiting_since"] = time.time()

    last_detector_frame_id = st.session_state.get("monitor_last_detector_frame_id", -1)
    if detector is not None and frame_id != last_detector_frame_id:
        try:
            detector.update_frame(frame, frame_id=frame_id)
        except TypeError:
            detector.update_frame(frame)
        st.session_state["monitor_last_detector_frame_id"] = frame_id
    detector_status = detector.get_status() if detector is not None and hasattr(detector, "get_status") else {}
    detector_last_output_at = detector_status.get("last_output_at")
    detector_last_error = detector_status.get("last_error")
    detector_last_processed_frame_id = detector_status.get("last_processed_frame_id", -1)
    detector_stale = (
        detector is not None
        and (
            detector_last_output_at in (None, 0.0)
            or (time.time() - float(detector_last_output_at)) > MONITOR_DETECTOR_STALE_SECONDS
            or detector_last_error
        )
    )

    if detector is not None and detector_stale and frame_id != detector_last_processed_frame_id:
        last_fallback_frame_id = st.session_state.get("monitor_last_sync_fallback_frame_id", -1)
        last_fallback_at = st.session_state.get("monitor_last_sync_fallback_at", 0.0)
        if frame_id != last_fallback_frame_id and (time.time() - last_fallback_at) > MONITOR_SYNC_FALLBACK_COOLDOWN:
            try:
                detector.force_process(frame, frame_id=frame_id)
                st.session_state["monitor_last_sync_fallback_frame_id"] = frame_id
                st.session_state["monitor_last_sync_fallback_at"] = time.time()
                detector_status = detector.get_status() if hasattr(detector, "get_status") else detector_status
                detector_last_error = detector_status.get("last_error")
            except Exception as exc:
                detector_last_error = repr(exc)
                st.session_state["monitor_last_sync_fallback_at"] = time.time()

    results, faces = detector.get_outputs() if detector is not None else ([], [])

    face_named = []
    detected_faces_count = 0
    recognized_students_now = set()
    if faces:
        detected_faces_count = len(faces)
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            name_face = identify_face(face, known_face_encodings_norm, known_face_names)
            face_named.append(((fx1, fy1, fx2, fy2), name_face))
            if name_face != "Desconhecido":
                recognized_students_now.add(name_face)
                remember_name((fx1, fy1, fx2, fy2), name_face)

    rendered_tracks_count = 0
    hidden_unknown_tracks_count = 0
    overlays = []
    rendered_identity_boxes = []

    if results:
        for result in results:
            if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
                continue
            keypoints_all = result.keypoints.data.cpu().numpy()
            hud_lines = []

            for pid, person_keypoints in enumerate(keypoints_all):
                if len(person_keypoints) == 0:
                    continue

                current_behavior = "Indeterminado"
                have_all = False
                pose_conf_threshold = _adaptive_keypoint_conf_threshold(person_keypoints)

                if person_keypoints.shape[0] > 10:
                    nose = person_keypoints[0]
                    ls, rs = person_keypoints[5], person_keypoints[6]
                    le, re = person_keypoints[7], person_keypoints[8]
                    lw, rw = person_keypoints[9], person_keypoints[10]

                    confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
                    have_all = all(c > pose_conf_threshold for c in confs)
                    if have_all:
                        current_behavior = classify_behavior(nose, ls, rs, le, re, lw, rw, pose_conf_threshold)

                x_coords = [p[0] for p in person_keypoints if p[2] > pose_conf_threshold]
                y_coords = [p[1] for p in person_keypoints if p[2] > pose_conf_threshold]
                if not x_coords or not y_coords:
                    continue
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                y_min = max(0, int(y_min - box_margin_ratio * (y_max - y_min)))
                person_box = (x_min, y_min, x_max, y_max)
                detector_person_box = _extract_detector_person_box(result, pid, frame.shape[:2])

                best_i, name_student = 0.0, "Desconhecido"
                for (fb, nm) in face_named:
                    i = iou(person_box, fb)
                    if i > best_i:
                        best_i, name_student = i, nm
                if best_i < 0.10:
                    name_student = resolve_name(person_box)

                behavior_key = name_student if name_student != "Desconhecido" else f"pid_{pid}"

                if person_keypoints.shape[0] > 10:
                    nose = person_keypoints[0]
                    l_eye = person_keypoints[1]
                    r_eye = person_keypoints[2]
                    l_ear = person_keypoints[3]
                    r_ear = person_keypoints[4]
                    ls = person_keypoints[5]
                    rs = person_keypoints[6]

                    if nose[2] > pose_conf_threshold or (ls[2] > pose_conf_threshold and rs[2] > pose_conf_threshold):
                        lateral_status = is_lateral_view(
                            nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=pose_conf_threshold
                        )
                        back_status = is_back_view(
                            nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=pose_conf_threshold
                        )
                        sleep_like_posture = False
                        if have_all:
                            sleep_metrics = _analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, pose_conf_threshold)
                            sleep_like_posture = sleep_metrics["strong_sleep"] or sleep_metrics["head_supported"]
                        new_behavior = check_distracted_status(
                            behavior_key,
                            (lateral_status or back_status) and not sleep_like_posture,
                            lateral_timers,
                            timeout=DISTRACTED_TIMEOUT_SECONDS,
                        )
                        if new_behavior:
                            current_behavior = new_behavior

                    raw_behavior = current_behavior
                    state = sleep_smoother.setdefault(
                        behavior_key,
                        {"state": "Indeterminado", "candidate": None, "candidate_count": 0},
                    )

                    if raw_behavior == state["state"]:
                        state["candidate"] = None
                        state["candidate_count"] = 0
                    else:
                        if raw_behavior == state["candidate"]:
                            state["candidate_count"] += 1
                        else:
                            state["candidate"] = raw_behavior
                            state["candidate_count"] = 1

                        required_frames = _transition_frames_required(state["state"], raw_behavior)
                        if state["candidate_count"] >= required_frames:
                            state["state"] = raw_behavior
                            state["candidate"] = None
                            state["candidate_count"] = 0

                    current_behavior = state["state"]

                display_box = _build_display_box(
                    person_keypoints,
                    current_behavior,
                    pose_conf_threshold,
                    fallback_box=detector_person_box or person_box,
                    frame_shape=frame.shape[:2],
                )
                if display_box is not None:
                    x_min, y_min, x_max, y_max = display_box

                if name_student != "Desconhecido" and episode_manager is not None:
                    recognized_students_now.add(name_student)
                    now_dt = get_local_now()
                    student_record = student_lookup.get(name_student, {})
                    episode_manager.update_behavior(
                        student_key=name_student,
                        student_name=name_student,
                        student_id=student_record.get("id"),
                        behavior=current_behavior,
                        timestamp=now_dt,
                        school=school,
                        discipline=discipline,
                        teacher=user_name,
                        source="realtime",
                    )

                if not should_render_track(name_student, show_unknown_boxes):
                    hidden_unknown_tracks_count += 1
                    continue

                is_negative_behavior = current_behavior in ("Agitado", "Dormindo", "Distraido")
                box_color_hex = "#ef4444" if is_negative_behavior else "#22c55e"
                label_bg_hex = "#991b1b" if is_negative_behavior else "#dcfce7"
                label_fg_hex = "#ffffff" if is_negative_behavior else "#14532d"
                label_text = f"{name_student} - {current_behavior}"
                rendered_tracks_count += 1
                rendered_identity_boxes.append((x_min, y_min, x_max, y_max))
                overlays.append(
                    {
                        "x1": x_min,
                        "y1": y_min,
                        "x2": x_max,
                        "y2": y_max,
                        "label": label_text,
                        "color": box_color_hex,
                        "label_bg": label_bg_hex,
                        "label_fg": label_fg_hex,
                    }
                )

                if show_debug and have_all:
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
                    y0 = 24
                    for i, text in enumerate(hud_lines):
                        overlays.append(
                            {
                                "x1": 10,
                                "y1": y0 + int(i * 22 * debug_font),
                                "x2": max(180, 10 + len(text) * 9),
                                "y2": y0 + int(i * 22 * debug_font) + 24,
                                "label": text,
                                "color": "#0ea5e9",
                            }
                        )

    last_rendered_frame_id = st.session_state.get("monitor_last_rendered_frame_id", -1)
    if frame_id != last_rendered_frame_id:
        st.session_state["monitor_last_rendered_frame_id"] = frame_id

    if detector_last_error:
        _update_monitor_status(
            status_placeholder,
            "warning",
            "A deteccao entrou em modo de recuperacao. "
            f"Ultimo erro do detector: {detector_last_error}",
        )
    elif known_face_encodings_norm is None or len(known_face_names) == 0:
        _update_monitor_status(
            status_placeholder,
            "warning",
            "Nenhum embedding de aluno foi carregado. "
            "Rode o processo de geracao de embeddings para habilitar o reconhecimento facial.",
        )
    else:
        _update_monitor_status(status_placeholder, "empty")

    recognized_faces_count = _update_recognized_students_metric(recognized_students_now)
    st.session_state["monitor_live_stats"] = {
        "detected_faces_count": detected_faces_count,
        "recognized_faces_count": recognized_faces_count,
        "rendered_tracks_count": rendered_tracks_count,
        "detector_last_error": detector_last_error,
    }
    render_monitor_video_summary_fragment(current_session_id=current_session_id, placeholder=summary_placeholder)
    st.session_state["monitor_last_overlays"] = overlays
    st.session_state["monitor_last_frame_shape"] = frame.shape[:2]
    _render_monitor_frame(frame_placeholder, frame, frame_id=frame_id, jpeg_bytes=frame_jpeg, overlays=overlays)
    return True

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


def _clamp_box_to_frame(box, frame_shape=None):
    x1, y1, x2, y2 = [float(v) for v in box]
    if frame_shape is not None:
        frame_h = max(1, int(frame_shape[0]))
        frame_w = max(1, int(frame_shape[1]))
        x1 = min(max(0.0, x1), float(frame_w - 1))
        y1 = min(max(0.0, y1), float(frame_h - 1))
        x2 = min(max(0.0, x2), float(frame_w))
        y2 = min(max(0.0, y2), float(frame_h))

    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0
    return tuple(int(round(v)) for v in (x1, y1, x2, y2))


def _extract_detector_person_box(result, person_index: int, frame_shape=None):
    boxes = getattr(result, "boxes", None)
    xyxy = None if boxes is None else getattr(boxes, "xyxy", None)
    if xyxy is None:
        return None
    try:
        total = int(xyxy.shape[0])
    except Exception:
        try:
            total = len(xyxy)
        except Exception:
            return None
    if person_index < 0 or person_index >= total:
        return None
    try:
        raw_box = xyxy[person_index]
        values = raw_box.tolist() if hasattr(raw_box, "tolist") else list(raw_box)
        x1, y1, x2, y2 = [float(v) for v in values[:4]]
    except Exception:
        return None
    return _clamp_box_to_frame((x1, y1, x2, y2), frame_shape)


def _visible_keypoint_xy(person_keypoints, idx: int, threshold: float):
    if idx < 0 or idx >= len(person_keypoints):
        return None
    point = person_keypoints[idx]
    if len(point) < 3 or float(point[2]) <= float(threshold):
        return None
    return (float(point[0]), float(point[1]))


def _build_display_box(person_keypoints, current_behavior: str, pose_conf_threshold: float, fallback_box=None, frame_shape=None):
    geometry_threshold = max(DISPLAY_BOX_MIN_KEYPOINT_CONF, float(pose_conf_threshold))
    nose = _visible_keypoint_xy(person_keypoints, 0, geometry_threshold)
    ls = _visible_keypoint_xy(person_keypoints, 5, geometry_threshold)
    rs = _visible_keypoint_xy(person_keypoints, 6, geometry_threshold)
    lh = _visible_keypoint_xy(person_keypoints, 11, geometry_threshold)
    rh = _visible_keypoint_xy(person_keypoints, 12, geometry_threshold)

    torso_points = [point for point in (nose, ls, rs, lh, rh) if point is not None]
    if len(torso_points) < 2:
        return _clamp_box_to_frame(fallback_box, frame_shape) if fallback_box is not None else None

    if ls is not None and rs is not None:
        center_x = (ls[0] + rs[0]) / 2.0
        shoulder_y = (ls[1] + rs[1]) / 2.0
        shoulder_span = max(DISPLAY_BOX_MIN_WIDTH, abs(rs[0] - ls[0]))
        torso_x_min = min(ls[0], rs[0])
        torso_x_max = max(ls[0], rs[0])
    else:
        xs = [point[0] for point in torso_points]
        ys = [point[1] for point in torso_points]
        center_x = sum(xs) / len(xs)
        shoulder_y = min(ys) + 0.35 * DISPLAY_BOX_MIN_WIDTH
        shoulder_span = max(DISPLAY_BOX_MIN_WIDTH, max(xs) - min(xs))
        torso_x_min = min(xs)
        torso_x_max = max(xs)

    hip_points = [point for point in (lh, rh) if point is not None]
    if hip_points:
        torso_x_min = min([torso_x_min] + [point[0] for point in hip_points])
        torso_x_max = max([torso_x_max] + [point[0] for point in hip_points])
        bottom = max(point[1] for point in hip_points) + 0.28 * shoulder_span
    else:
        bottom = shoulder_y + 2.15 * shoulder_span

    if nose is not None:
        top = nose[1] - 0.55 * shoulder_span
    else:
        top = shoulder_y - 0.95 * shoulder_span

    half_width = max(
        0.82 * shoulder_span,
        0.72 * max(1.0, torso_x_max - torso_x_min),
        DISPLAY_BOX_MIN_WIDTH / 2.0,
    )
    left = center_x - half_width
    right = center_x + half_width

    if current_behavior in ("Perguntando", "Agitado"):
        shoulder_ceiling = min(point[1] for point in (ls, rs) if point is not None) if (ls or rs) else shoulder_y
        arm_points = []
        for idx in (7, 8, 9, 10):
            point = _visible_keypoint_xy(person_keypoints, idx, geometry_threshold)
            if point is None:
                continue
            if point[1] < shoulder_ceiling + 0.55 * shoulder_span:
                arm_points.append(point)
        if arm_points:
            left = min(left, min(point[0] for point in arm_points) - 0.18 * shoulder_span)
            right = max(right, max(point[0] for point in arm_points) + 0.18 * shoulder_span)
            top = min(top, min(point[1] for point in arm_points) - 0.18 * shoulder_span)

    return _clamp_box_to_frame((left, top, right, bottom), frame_shape)


def should_render_track(identity: str, show_unknown_boxes: bool = False) -> bool:
    if show_unknown_boxes:
        return True
    normalized_identity = (identity or "").strip().lower()
    return normalized_identity not in UNKNOWN_IDENTITY_LABELS


def identify_face(face, known_face_encodings_norm, known_face_names):
    if known_face_encodings_norm is None or len(known_face_names) == 0:
        return "Desconhecido"

    emb = face.embedding
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    sims = np.dot(known_face_encodings_norm, emb)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    second_best_score = -1.0
    if len(sims) > 1:
        second_best_score = float(np.partition(sims, -2)[-2])

    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    if face_area < 3500:
        acceptance_threshold = FACE_RECOGNITION_TINY_THRESHOLD
        min_margin = FACE_RECOGNITION_TINY_MARGIN
    elif face_area < 7000:
        acceptance_threshold = FACE_RECOGNITION_SMALL_THRESHOLD
        min_margin = FACE_RECOGNITION_MIN_MARGIN * 0.7
    elif face_area < 14000:
        acceptance_threshold = FACE_RECOGNITION_MEDIUM_THRESHOLD
        min_margin = FACE_RECOGNITION_MIN_MARGIN
    else:
        acceptance_threshold = FACE_RECOGNITION_BASE_THRESHOLD
        min_margin = FACE_RECOGNITION_MIN_MARGIN

    margin = best_score - second_best_score if second_best_score >= 0 else best_score
    if best_score >= acceptance_threshold and margin >= min_margin:
        return known_face_names[best_idx]

    return "Desconhecido"


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


def _merge_face_detections(base_faces, extra_faces, iou_threshold: float = FAR_FACE_MERGE_IOU):
    merged_faces = list(base_faces or [])
    for face in extra_faces or []:
        face_box = tuple(float(v) for v in face.bbox.astype(float))
        if any(iou(face_box, tuple(float(v) for v in existing.bbox.astype(float))) >= iou_threshold for existing in merged_faces):
            continue
        merged_faces.append(face)
    return merged_faces

# ---------------- Detector em thread separada (IA fora do loop de render) ----------------
class DetectorWorker:
    """
    Roda YOLO (pose) + InsightFace em background, sempre no frame mais recente.
    Evita fila e mantém o vídeo "ao vivo".
    """
    def __init__(
        self,
        model_pose,
        model_face,
        device,
        min_inference_interval=0.18,
        face_refresh_interval=0.20,
        pose_imgsz=960,
        pose_conf=0.22,
    ):
        self.model_pose = model_pose
        self.model_face = model_face
        self.device = device
        self.pose_imgsz = int(pose_imgsz)
        self.pose_conf = float(pose_conf)
        self._latest_frame = None
        self._latest_frame_id = -1
        self._last_results = []
        self._last_faces = []
        self._lock = threading.Lock()
        self._running = False
        self._th = None
        self._last_inference_at = 0.0
        self._min_inference_interval = float(min_inference_interval)
        self._face_refresh_interval = float(face_refresh_interval)
        self._last_face_inference_at = 0.0
        self._last_output_at = 0.0
        self._last_error = None
        self._last_processed_frame_id = -1
        self._processed_frames = 0
        self._fallback_mode = None

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

    def update_frame(self, frame, frame_id=None):
        with self._lock:
            if frame_id is not None and frame_id <= self._latest_frame_id:
                return
            self._latest_frame = frame.copy()
            if frame_id is None:
                self._latest_frame_id += 1
            else:
                self._latest_frame_id = frame_id

    def get_outputs(self):
        with self._lock:
            faces = self._last_faces
            results = self._last_results
        return results, faces

    def get_status(self):
        with self._lock:
            return {
                "last_output_at": self._last_output_at,
                "last_error": self._last_error,
                "last_processed_frame_id": self._last_processed_frame_id,
                "processed_frames": self._processed_frames,
                "device": self.device,
                "fallback_mode": self._fallback_mode,
            }

    def _switch_to_cpu(self, reason: Exception | str | None = None):
        self.model_pose = _create_pose_model("cpu")
        self.model_face = _create_face_model("cpu")
        self.device = "cpu"
        self.pose_imgsz = POSE_IMGSZ_CPU
        self._face_refresh_interval = FACE_REFRESH_INTERVAL_CPU
        self._fallback_mode = "cpu"
        self._last_error = None if reason is None else f"Detector mudou para CPU: {reason}"

    def _detect_faces(self, frame_rgb):
        base_faces = list(self.model_face.get(frame_rgb))
        if (
            FAR_FACE_UPSCALE <= 1.0
            or len(base_faces) > FAR_FACE_EXTRA_PASS_MAX_BASE_FACES
        ):
            return base_faces

        def _run_scaled_face_pass(source_rgb, y_offset: int = 0):
            upscaled = cv2.resize(
                source_rgb,
                None,
                fx=FAR_FACE_UPSCALE,
                fy=FAR_FACE_UPSCALE,
                interpolation=cv2.INTER_CUBIC,
            )
            detected_faces = list(self.model_face.get(upscaled))
            if not detected_faces:
                return []

            scale = float(FAR_FACE_UPSCALE)
            for face in detected_faces:
                bbox = face.bbox.astype(np.float32)
                bbox[0] /= scale
                bbox[2] /= scale
                bbox[1] = (bbox[1] / scale) + y_offset
                bbox[3] = (bbox[3] / scale) + y_offset
                face.bbox = bbox
            return detected_faces

        frame_h, frame_w = frame_rgb.shape[:2]
        crop_h = max(1, int(frame_h * FAR_FACE_REGION_TOP_RATIO))
        if crop_h >= frame_h:
            crop_rgb = frame_rgb
            y_offset = 0
        else:
            crop_rgb = frame_rgb[:crop_h, :, :]
            y_offset = 0

        boosted_faces = _run_scaled_face_pass(crop_rgb, y_offset=y_offset)
        if not boosted_faces and not base_faces and crop_h < frame_h:
            boosted_faces = _run_scaled_face_pass(frame_rgb, y_offset=0)
        if not boosted_faces:
            return base_faces

        return _merge_face_detections(base_faces, boosted_faces)

    def _infer(self, frame):
        try:
            results = self.model_pose.predict(
                frame,
                show=False,
                device=self.device,
                verbose=False,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
                half=(self.device == "cuda"),
            )
            now = time.time()
            should_refresh_faces = (
                not self._last_faces
                or (now - self._last_face_inference_at) >= self._face_refresh_interval
            )
            if should_refresh_faces:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self._detect_faces(rgb)
                self._last_face_inference_at = now
            else:
                faces = self._last_faces
            return results, faces
        except Exception as exc:
            if self.device == "cuda" and _is_cuda_runtime_error(exc):
                self._switch_to_cpu(reason=exc)
                results = self.model_pose.predict(
                    frame,
                    show=False,
                    device=self.device,
                    verbose=False,
                    imgsz=self.pose_imgsz,
                    conf=self.pose_conf,
                    half=False,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self._detect_faces(rgb)
                self._last_face_inference_at = time.time()
                return results, faces
            raise

    def force_process(self, frame, frame_id=None):
        results, faces = self._infer(frame)
        with self._lock:
            self._last_faces = faces
            self._last_results = results
            self._last_output_at = time.time()
            self._last_error = None
            self._processed_frames += 1
            if frame_id is not None:
                self._last_processed_frame_id = frame_id
                if frame_id == self._latest_frame_id:
                    self._latest_frame = None
        self._last_inference_at = time.time()
        return results, faces

    def _run(self):
        while self._running:
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                frame_id = self._latest_frame_id
            if frame is None:
                time.sleep(0.003)
                continue
            if frame_id < 0:
                time.sleep(0.003)
                continue
            now = time.time()
            if now - self._last_inference_at < self._min_inference_interval:
                time.sleep(0.005)
                continue
            try:
                results, faces = self._infer(frame)
                with self._lock:
                    self._last_faces = faces
                    self._last_results = results
                    self._last_output_at = time.time()
                    self._last_error = None
                    self._last_processed_frame_id = frame_id
                    self._processed_frames += 1
                    if frame_id == self._latest_frame_id:
                        self._latest_frame = None
                self._last_inference_at = time.time()
            except Exception as exc:
                with self._lock:
                    self._last_error = repr(exc)
                time.sleep(0.05)

# ---------------- Funções auxiliares de comportamento ----------------
def is_lateral_view(nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=0.5, cam_side="LEFT", cam_offset=0.12):
   
    # escala da pessoa
    s = float(abs(ls[0] - rs[0])) + 1e-6

    # limiar dinâmico p/ razão de deslocamento (nariz vs olhos)
    if s < 40:
        t_ratio = 0.18
    elif s < 60:
        t_ratio = 0.24
    else:
        t_ratio = 0.30

    # offset conforme o lado da camera

    if cam_side == "LEFT":
        offset = -cam_offset
    elif cam_side == "RIGHT":
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
        (l_eye[2] < conf_thr and r_eye[2] > conf_thr + 0.08) or
        (r_eye[2] < conf_thr and l_eye[2] > conf_thr + 0.08)
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
            (l_ear[2] < conf_thr and r_ear[2] > conf_thr + 0.08) or
            (r_ear[2] < conf_thr and l_ear[2] > conf_thr + 0.08)
        )
        cond_ears = cond_ratio_ears or cond_conf_ears

    return cond_ratio_eyes or cond_conf_eyes or cond_ears

def is_back_view(nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=0.5):
    shoulder_width = float(abs(ls[0] - rs[0]))
    if ls[2] <= conf_thr or rs[2] <= conf_thr or shoulder_width < 35.0:
        return False

    frontal_face_missing = (
        nose[2] < conf_thr and
        l_eye[2] < conf_thr and
        r_eye[2] < conf_thr
    )
    ears_missing = l_ear[2] < conf_thr and r_ear[2] < conf_thr
    shoulder_balance = abs(ls[1] - rs[1]) < max(18.0, 0.35 * shoulder_width)

    return frontal_face_missing and ears_missing and shoulder_balance

def check_distracted_status(name, is_distracted_pose, lateral_timers, timeout=10):
    now = time.time()
    if name not in lateral_timers:
        lateral_timers[name] = {"start_time": None, "is_lateral": False}
    if is_distracted_pose:
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

def _point_distance(p1, p2) -> float:
    return float(np.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


def _analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, threshold):
    cx = (ls[0] + rs[0]) / 2.0
    cy = (ls[1] + rs[1]) / 2.0
    s = max(1.0, float(abs(ls[0] - rs[0])))

    left_wrist_visible = lw[2] > threshold
    right_wrist_visible = rw[2] > threshold
    left_elbow_visible = le[2] > threshold
    right_elbow_visible = re[2] > threshold

    nose_below_shoulders = nose[1] > cy + 0.02 * s
    nose_far_below_shoulders = nose[1] > cy + 0.12 * s
    nose_centered = abs(nose[0] - cx) < 0.40 * s

    left_wrist_near_nose = left_wrist_visible and (
        _point_distance(lw, nose) < 0.38 * s
        or (abs(lw[0] - nose[0]) < 0.28 * s and abs(lw[1] - nose[1]) < 0.32 * s)
    )
    right_wrist_near_nose = right_wrist_visible and (
        _point_distance(rw, nose) < 0.38 * s
        or (abs(rw[0] - nose[0]) < 0.28 * s and abs(rw[1] - nose[1]) < 0.32 * s)
    )
    left_elbow_near_nose = left_elbow_visible and (
        _point_distance(le, nose) < 0.44 * s
        or (abs(le[0] - nose[0]) < 0.34 * s and abs(le[1] - nose[1]) < 0.26 * s)
    )
    right_elbow_near_nose = right_elbow_visible and (
        _point_distance(re, nose) < 0.44 * s
        or (abs(re[0] - nose[0]) < 0.34 * s and abs(re[1] - nose[1]) < 0.26 * s)
    )

    left_arm_support = (
        left_wrist_visible
        and left_elbow_visible
        and abs(lw[0] - le[0]) < 0.42 * s
        and abs(lw[1] - le[1]) < 0.34 * s
    )
    right_arm_support = (
        right_wrist_visible
        and right_elbow_visible
        and abs(rw[0] - re[0]) < 0.42 * s
        and abs(rw[1] - re[1]) < 0.34 * s
    )

    wrists_below_shoulders = sum(
        1
        for wrist, visible in ((lw, left_wrist_visible), (rw, right_wrist_visible))
        if visible and wrist[1] > cy - 0.02 * s
    )
    elbows_below_shoulders = sum(
        1
        for elbow, visible in ((le, left_elbow_visible), (re, right_elbow_visible))
        if visible and elbow[1] > cy - 0.08 * s
    )

    wrist_near_count = int(left_wrist_near_nose) + int(right_wrist_near_nose)
    elbow_near_count = int(left_elbow_near_nose) + int(right_elbow_near_nose)
    support_count = int(left_arm_support) + int(right_arm_support)

    head_supported = (
        (left_wrist_near_nose and (left_arm_support or left_elbow_near_nose))
        or (right_wrist_near_nose and (right_arm_support or right_elbow_near_nose))
        or (wrist_near_count >= 1 and elbow_near_count >= 1)
    )

    moderate_sleep = (
        nose_below_shoulders
        and wrists_below_shoulders >= 1
        and elbows_below_shoulders >= 1
        and (
            head_supported
            or (wrist_near_count >= 1 and nose_centered)
            or (elbow_near_count >= 1 and nose_far_below_shoulders)
            or (support_count >= 1 and nose_far_below_shoulders)
        )
    )

    strong_sleep = (
        nose_below_shoulders
        and nose_centered
        and wrists_below_shoulders >= 1
        and elbows_below_shoulders >= 1
        and (
            (wrist_near_count >= 1 and elbow_near_count >= 1)
            or (wrist_near_count >= 1 and support_count >= 1)
            or (nose_far_below_shoulders and elbow_near_count >= 1)
        )
    )

    return {
        "scale": s,
        "nose_below_shoulders": nose_below_shoulders,
        "nose_far_below_shoulders": nose_far_below_shoulders,
        "nose_centered": nose_centered,
        "wrist_near_count": wrist_near_count,
        "elbow_near_count": elbow_near_count,
        "support_count": support_count,
        "wrists_below_shoulders": wrists_below_shoulders,
        "elbows_below_shoulders": elbows_below_shoulders,
        "head_supported": head_supported,
        "moderate_sleep": moderate_sleep,
        "strong_sleep": strong_sleep,
    }


def _is_natural_distracted_pose(nose, ls, rs, le, re, lw, rw, threshold):
    sleep_metrics = _analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, threshold)
    s = sleep_metrics["scale"]
    cy = (ls[1] + rs[1]) / 2.0
    shoulder_tilt = abs(ls[1] - rs[1])
    nose_depth = nose[1] - cy
    elbows_high = sum(
        1
        for elbow in (le, re)
        if elbow[2] > threshold and elbow[1] < cy - 0.22 * s
    )
    arms_relaxed = (
        sleep_metrics["wrist_near_count"] == 0
        and sleep_metrics["elbow_near_count"] == 0
        and sleep_metrics["support_count"] == 0
    )

    return (
        not sleep_metrics["strong_sleep"]
        and not sleep_metrics["moderate_sleep"]
        and not sleep_metrics["head_supported"]
        and arms_relaxed
        and nose_depth < 0.18 * s
        and shoulder_tilt < max(24.0, 0.26 * s)
        and elbows_high == 0
    )


def _transition_frames_required(current_state: str, candidate_state: str) -> int:
    if current_state == "Dormindo" and candidate_state != "Dormindo":
        return EXIT_SLEEP_FRAMES

    return {
        "Dormindo": ENTER_SLEEP_FRAMES,
        "Perguntando": ENTER_QUESTION_FRAMES,
        "Atento": ENTER_ATTENTIVE_FRAMES,
        "Agitado": ENTER_AGITATED_FRAMES,
        "Distraido": ENTER_DISTRACTED_FRAMES,
        "Indeterminado": ENTER_UNDETERMINED_FRAMES,
    }.get(candidate_state, ENTER_UNDETERMINED_FRAMES)


def classify_behavior(nose, ls, rs, le, re, lw, rw, threshold):
    """
    0:nose | 5-6: ombros (ls, rs) | 7-8: cotovelos (le, re) | 9-10: punhos (lw, rw)
    """
    cx = (ls[0] + rs[0]) / 2.0
    cy = (ls[1] + rs[1]) / 2.0
    s = max(1.0, float(abs(ls[0] - rs[0])))   # escala ombro-a-ombro
    head_clearance = cy - nose[1]
    head_offset_x = abs(nose[0] - cx)
    shoulder_tilt = abs(ls[1] - rs[1])
    sleep_metrics = _analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, threshold)

    # "Cabeça baixa" bloqueia "Atento" e prioriza estados conservadores.
    is_head_low = head_clearance < 0.16 * s or nose[1] > cy - 0.03 * s
    is_looking_down = head_clearance < 0.10 * s or nose[1] > cy + 0.06 * s

    left_hand_near_head = (
        lw[2] > threshold and (
            _point_distance(lw, nose) < 0.34 * s or
            (abs(lw[0] - nose[0]) < 0.24 * s and abs(lw[1] - nose[1]) < 0.30 * s)
        )
    )
    right_hand_near_head = (
        rw[2] > threshold and (
            _point_distance(rw, nose) < 0.34 * s or
            (abs(rw[0] - nose[0]) < 0.24 * s and abs(rw[1] - nose[1]) < 0.30 * s)
        )
    )

    # "Dormindo" exige assinatura forte de apoio: nariz abaixo da linha dos ombros
    # e proximidade consistente com punho/cotovelo, evitando confundir cabeça baixa
    # simples com distração lateral.
    if s >= 28.0 and (sleep_metrics["strong_sleep"] or sleep_metrics["moderate_sleep"]):
        return "Dormindo"

    if is_looking_down:
        return "Indeterminado"
    if is_head_low:
        return "Indeterminado"

    # --- MÃOS ALTAS -> Perguntando/Agitado ---
    # "Perguntando" agora exige mão realmente erguida, afastada da cabeça
    # e depois ainda passa por persistência temporal para evitar falso positivo.
    raised_margin = 0.22 * s
    min_head_gap_x = 0.28 * s
    elbow_margin = 0.04 * s
    up_L = (
        lw[2] > threshold and le[2] > threshold and
        lw[1] < cy - raised_margin and
        lw[1] < le[1] - elbow_margin and
        abs(lw[0] - nose[0]) > min_head_gap_x and
        not left_hand_near_head
    )
    up_R = (
        rw[2] > threshold and re[2] > threshold and
        rw[1] < cy - raised_margin and
        rw[1] < re[1] - elbow_margin and
        abs(rw[0] - nose[0]) > min_head_gap_x and
        not right_hand_near_head
    )
    if up_L and up_R:
        return "Agitado" if abs(lw[0] - rw[0]) > 0.95 * s else "Perguntando"
    if up_L or up_R:
        return "Perguntando"

    # --- ATENTO ---
    # "Atento" precisa ser conquistado: cabeça suficientemente acima dos ombros,
    # sem sinal de cabeça baixa, sem mão apoiada no rosto e sem postura ambígua.
    hands_compact_front = (
        lw[2] > threshold and rw[2] > threshold and
        abs(lw[0] - rw[0]) < 0.45 * s and
        min(lw[1], rw[1]) > nose[1] and
        max(lw[1], rw[1]) < cy + 0.45 * s
    )
    phone_like_posture = hands_compact_front and head_clearance < 0.28 * s
    attentive_posture = (
        head_clearance > 0.24 * s and
        head_offset_x < 0.34 * s and
        shoulder_tilt < max(16.0, 0.18 * s) and
        not phone_like_posture and
        not left_hand_near_head and
        not right_hand_near_head
    )
    if attentive_posture:
        return "Atento"

    # Ambiguidade deixa de cair em "Atento" por padrão.
    return "Indeterminado"


def _adaptive_keypoint_conf_threshold(person_keypoints) -> float:
    visible_points = [p for p in person_keypoints if len(p) >= 3 and float(p[2]) > 0.08]
    if not visible_points:
        return 0.22

    xs = [float(p[0]) for p in visible_points]
    ys = [float(p[1]) for p in visible_points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    size = max(width, height)

    if size < 110:
        return 0.14
    if size < 180:
        return 0.17
    if size < 260:
        return 0.19
    return 0.22


# ------------------ CRIPTOGRAFAR NOMES ---------------------------
def criptografar_nome_matricula(nome, matricula):
    return hashlib.sha256(f"{nome}_{matricula}".encode()).hexdigest()

# ------------------------------ APP ------------------------------
def recognition_behavior():
    school = DEFAULT_SCHOOL_NAME

    _inject_sidebar_menu_styles()
    user_context = st.session_state.get("user_context") or {}
    user_name = user_context.get("name", st.session_state.get("name", "Usuário"))
    user_role = user_context.get("role", st.session_state.get("role", "professor"))
    profile_role_label = "Professor(a)" if user_role == "professor" else "Administrador(a)"
    teacher_menu = ["Cadastro de Alunos", "Monitoramento", "Gráficos", "Relatórios"]
    admin_menu = ["Usuários", "Gráficos", "Relatórios"]
    raw_menu_options = admin_menu if user_role == "admin" else teacher_menu
    menu_labels = {
        "Cadastro de Alunos": "Cadastro de Alunos\nGerencie alunos e cadastros",
        "Monitoramento": "Monitoramento\nAcompanhe as sessões em tempo real",
        "Gráficos": "Gráficos\nVisualize dados e estatísticas",
        "Relatórios": "Relatórios\nAcesse análises e relatórios observacionais",
        "Usuários": "🛠️    Usuários\nGerencie contas e manutenção do sistema",
    }
    menu_display_options = [menu_labels[option] for option in raw_menu_options]
    menu_widget_key = "sidebar_menu_display"
    if st.session_state.get(menu_widget_key) not in menu_display_options:
        st.session_state[menu_widget_key] = menu_labels[raw_menu_options[0]]

    st.sidebar.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<img class='sidebar-hero-img' src='data:image/png;base64,{img_to_base64(image_path_classroom)}' alt='Sala de aula' />",
        unsafe_allow_html=True,
    )
    profile_col, logout_col = st.sidebar.columns([4, 1.2])
    with profile_col:
        st.markdown(
            f"""
            <div class="sidebar-profile">
                <div class="sidebar-avatar">🎓</div>
                <div>
                    <p class="sidebar-profile-name">{html.escape(user_name)}</p>
                    <p class="sidebar-profile-role">{html.escape(profile_role_label)}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with logout_col:
        logout_clicked = st.button("↩", key="sidebar_logout_button", use_container_width=True)
    st.sidebar.markdown("<div class='sidebar-profile-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-menu-label'>Menu</div>", unsafe_allow_html=True)
    menu_display_selected = st.sidebar.radio(
        "Menu",
        menu_display_options,
        key=menu_widget_key,
        label_visibility="collapsed",
    )
    menu_option = next(key for key, value in menu_labels.items() if value == menu_display_selected)
    st.session_state["current_menu_option"] = menu_option

    if logout_clicked:
        st.session_state[AUTH_RESTORE_BLOCK_KEY] = True
        st.session_state[AUTH_BOOTSTRAP_KEY] = True
        st.session_state[AUTH_BOOTSTRAP_STARTED_AT_KEY] = 0.0
        cookie_controller = CookieController(key="auth_cookies")
        try:
            if isinstance(cookie_controller.getAll(), dict) and cookie_controller.get(AUTH_COOKIE_NAME) is not None:
                cookie_controller.remove(AUTH_COOKIE_NAME, path="/", same_site="strict")
        except Exception:
            pass
        st.session_state["authenticated"] = False
        st.session_state["cpf"] = None
        st.session_state["city"] = None
        st.session_state["state"] = None
        st.session_state["name"] = None
        st.session_state["role"] = None
        st.session_state.pop("user_context", None)
        for key in LEGACY_AUTH_QUERY_KEYS:
            if key in st.query_params:
                del st.query_params[key]
        if AUTH_QUERY_TOKEN_KEY in st.query_params:
            del st.query_params[AUTH_QUERY_TOKEN_KEY]
        for key in (
            "monitoring_state",
            "last_closed_monitoring_session",
            "episode_manager",
            "student_lookup",
            "monitor_runtime",
            "monitor_waiting_since",
            "monitor_last_display_frame",
            "monitor_last_detector_frame_id",
            "monitor_last_rendered_frame_id",
            "monitor_last_sync_fallback_frame_id",
            "monitor_last_sync_fallback_at",
            "monitor_last_overlays",
            "monitor_last_frame_shape",
            "current_menu_option",
            "monitor_selected_subject_label",
            "monitor_selected_class_label",
            "monitor_selected_lesson_type",
        ):
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.markdown(
        """
        <div class="sidebar-help-box">
            <div class="sidebar-help-row">
                <div class="sidebar-help-icon">i</div>
                <div>
                    <div class="sidebar-help-title">Plataforma de Monitoramento Comportamental</div>
                    <div class="sidebar-help-subtitle">Turmas A • IA-2026</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # ------------------ CADASTRO ------------------
    if menu_option == "Cadastro de Alunos":
        _inject_student_registration_styles()
        _render_student_registration_header()
        allowed_classes_df = list_classes_for_user(user_context)
        if allowed_classes_df.empty:
            st.warning("Nenhuma turma vinculada ao professor foi encontrada. Cadastre os vínculos no banco antes de usar esta tela.")
            return

        # Parâmetros da captura automática
        IMAGENS_POR_POSE = 10
        capture_interval = 0.8
        prep_seconds = 2

        POSES = ["frontal", "lateral_direita", "lateral_esquerda", "cabeca_baixa"]

        # Estado
        pose_index        = st.session_state.get("pose_index", 0)
        img_index         = st.session_state.get("img_index", 0)
        cap_running       = st.session_state.get("cap_running", False)
        next_time         = st.session_state.get("next_time", None)
        pose_done         = st.session_state.get("pose_done", False)
        registration_done = st.session_state.get("registration_done", False)

        def _sync_student_mapping(student_id: str, student_name: str, student_registration: str) -> None:
            if not student_id or not student_name or not student_registration:
                return

            if not os.path.exists(MAPPING_CSV):
                pd.DataFrame(columns=["nome", "matricula", "hash"]).to_csv(MAPPING_CSV, index=False)

            df = pd.read_csv(MAPPING_CSV)
            hash_mask = df["hash"].astype(str).str.strip() == str(student_id).strip()
            identity_mask = (
                df["nome"].astype(str).str.strip() == str(student_name).strip()
            ) & (
                df["matricula"].astype(str).str.strip() == str(student_registration).strip()
            )

            if hash_mask.any():
                df.loc[hash_mask, ["nome", "matricula", "hash"]] = [student_name, student_registration, student_id]
            elif identity_mask.any():
                df.loc[identity_mask, "hash"] = student_id
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [{"nome": student_name, "matricula": student_registration, "hash": student_id}]
                        ),
                    ],
                    ignore_index=True,
                )

            df = df.drop_duplicates(subset=["hash"], keep="first")
            df = df.drop_duplicates(subset=["nome", "matricula"], keep="first")
            df.to_csv(MAPPING_CSV, index=False)

        def _persist_completed_student_registration(
            student_id: str,
            student_name: str,
            student_registration: str,
            class_id: int | None,
        ) -> str:
            if not student_id or not student_name or not student_registration:
                raise ValueError("Dados insuficientes para persistir o cadastro do aluno.")

            upsert_student(student_id, student_name, student_registration, class_id)
            _sync_student_mapping(student_id, student_name, student_registration)
            return student_id

        # Tela de conclusão
        if registration_done:
            ultimo_nome = st.session_state.get("last_cad_nome", "")
            ultima_mat  = st.session_state.get("last_cad_matricula", "")
            ultimo_hash = st.session_state.get("last_cad_hash", "")
            embedding_status = st.session_state.get("last_embedding_status")
            embedding_message = st.session_state.get("last_embedding_message", "")

            if ultimo_hash and embedding_status is None:
                progress_status = st.empty()
                progress_bar = st.progress(0, text="Finalizando cadastro do aluno...")
                progress_status.info("Preparando o registro do aluno e organizando as imagens capturadas.")
                progress_bar.progress(20, text="Validando dados e preparando o processamento facial...")
                progress_status.info("Gerando o embedding facial e registrando o aluno na base.")
                progress_bar.progress(55, text="Gerando embedding facial do aluno...")
                success, message = generate_student_embedding(
                    ultimo_hash,
                    ultimo_nome,
                    ultima_mat or None,
                )
                progress_bar.progress(90, text="Atualizando dados do monitoramento...")
                st.session_state["last_embedding_status"] = success
                st.session_state["last_embedding_message"] = message
                teardown_monitor_runtime()
                st.session_state.pop("student_lookup", None)
                if success:
                    progress_status.success("Cadastro e processamento facial concluídos.")
                    progress_bar.progress(100, text="Aluno registrado com sucesso.")
                else:
                    progress_status.error("O cadastro foi concluído, mas houve falha no processamento facial.")
                    progress_bar.progress(100, text="Cadastro concluído com pendência no embedding.")
                st.rerun()

            if ultimo_nome or ultima_mat:
                st.success(f"✅ Cadastro concluído para **{ultimo_nome}** (Matrícula **{ultima_mat}**).")
            else:
                st.success("✅ Cadastro concluído.")

            if embedding_status:
                st.success(embedding_message or "Embedding facial gerado com sucesso.")
            elif embedding_status is False:
                st.error(embedding_message or "Não foi possível gerar o embedding facial do aluno.")

            if ultimo_hash and st.button("🔄 Reprocessar embedding deste aluno", use_container_width=False):
                with st.spinner("Reprocessando embedding facial do aluno..."):
                    success, message = generate_student_embedding(
                        ultimo_hash,
                        ultimo_nome,
                        ultima_mat or None,
                    )
                st.session_state["last_embedding_status"] = success
                st.session_state["last_embedding_message"] = message
                teardown_monitor_runtime()
                st.session_state.pop("student_lookup", None)
                st.rerun()

            if st.button("✅ Finalizar cadastro"):
                _release_registration_camera()

                for k in ["pose_index","img_index","cap_running","next_time","pose_done",
                          "registration_done","last_cad_nome","last_cad_matricula","last_cad_hash",
                          "last_embedding_status","last_embedding_message","last_browser_capture_digest"]:
                    st.session_state.pop(k, None)
                st.session_state["student_registration_stage"] = "identify"
                st.session_state.pop("cad_nome_locked", None)
                st.session_state.pop("cad_matricula_locked", None)
                st.session_state.pop("cad_class_id_locked", None)

                st.session_state.cad_nome = ""
                st.session_state.cad_matricula = ""

                st.toast("Cadastro finalizado.")
                st.rerun()
            st.stop()

        pose_index = max(0, min(pose_index, len(POSES) - 1))
        pose_atual = POSES[pose_index]
        class_options = {
            int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
            for _, row in allowed_classes_df.iterrows()
        }
        class_option_ids = list(class_options.keys())
        if "cad_class_id" not in st.session_state and class_option_ids:
            st.session_state["cad_class_id"] = class_option_ids[0]
        if st.session_state.get("cad_class_id") not in class_options:
            st.session_state["cad_class_id"] = class_option_ids[0] if class_option_ids else None

        selected_class_id_for_students = st.session_state.get("cad_class_id")
        selected_class_students_df = list_students_for_user(user_context, selected_class_id_for_students)
        selected_class_students = selected_class_students_df.to_dict(orient="records")
        existing_student_options = {
            str(row["id"]): _format_student_option(row)
            for row in selected_class_students
        }
        existing_student_ids = list(existing_student_options.keys())
        if "cad_student_mode" not in st.session_state:
            st.session_state["cad_student_mode"] = "existing" if existing_student_ids else "new"
        if st.session_state.get("cad_student_mode") == "existing" and not existing_student_ids:
            st.session_state["cad_student_mode"] = "new"
        if st.session_state.get("cad_existing_student_id") not in existing_student_options:
            st.session_state["cad_existing_student_id"] = existing_student_ids[0] if existing_student_ids else None
        current_selected_existing_student = next(
            (
                row
                for row in selected_class_students
                if str(row["id"]) == str(st.session_state.get("cad_existing_student_id"))
            ),
            None,
        )
        generated_matricula = get_next_student_registration()

        nome_aluno = st.session_state.get("cad_nome", "")
        matricula = st.session_state.get("cad_matricula", "")
        locked_nome = st.session_state.get("cad_nome_locked", "")
        locked_matricula = st.session_state.get("cad_matricula_locked", "")
        locked_class_id = st.session_state.get("cad_class_id_locked")
        locked_student_mode = st.session_state.get("cad_student_mode_locked")
        locked_existing_student_id = st.session_state.get("cad_existing_student_id_locked")
        nome_norm = _normalize_student_name(nome_aluno)
        matr_norm = _normalize_student_registration(matricula)
        should_validate_registration = not (
            st.session_state.get("cad_student_mode") == "existing" and current_selected_existing_student is not None
        )
        registration_errors = (
            _validate_student_registration_fields(
                nome_aluno,
                matricula,
                require_matricula=st.session_state.get("cad_student_mode") != "new",
            )
            if should_validate_registration and (
                bool(nome_aluno)
                if st.session_state.get("cad_student_mode") == "new"
                else bool(nome_aluno or matricula)
            )
            else []
        )
        registration_ready = bool(nome_norm and matr_norm and not registration_errors and st.session_state.get("cad_class_id") is not None)

        registration_stage = st.session_state.get("student_registration_stage", "identify")
        if registration_stage == "capture" and locked_nome and locked_matricula and locked_class_id is not None:
            nome_aluno = locked_nome
            matricula = locked_matricula
            nome_norm = _normalize_student_name(locked_nome)
            matr_norm = _normalize_student_registration(locked_matricula)
            registration_errors = (
                []
                if locked_student_mode == "existing"
                else _validate_student_registration_fields(locked_nome, locked_matricula)
            )
            registration_ready = bool(nome_norm and matr_norm and not registration_errors and locked_class_id in class_options)
        if not registration_ready and registration_stage == "capture":
            registration_stage = "identify"
            st.session_state["student_registration_stage"] = registration_stage

        start_btn = False
        cancel_btn = False
        next_btn = False
        nome_criptografado = ""
        pasta_base = ""

        if registration_stage == "identify":
            with st.container(border=True):
                _render_student_registration_card_header(
                    "👥",
                    "Dados do Aluno",
                    "Preencha os dados principais para iniciar o cadastro.",
                    "linear-gradient(180deg, #FFFFFF 0%, #EAF4FF 55%, #D7E9FF 100%)",
                )
                _render_student_registration_stepper(1)
                st.markdown(
                    "<div class='student-reg-info'>Selecione a turma e informe os dados do aluno para iniciar o cadastro.</div>",
                    unsafe_allow_html=True,
                )

                st.selectbox(
                    "Turma",
                    class_option_ids,
                    key="cad_class_id",
                    format_func=lambda value: class_options.get(value, ""),
                )

                selected_class_id_for_students = st.session_state.get("cad_class_id")
                selected_class_students_df = list_students_for_user(user_context, selected_class_id_for_students)
                selected_class_students = selected_class_students_df.to_dict(orient="records")
                existing_student_options = {
                    str(row["id"]): _format_student_option(row)
                    for row in selected_class_students
                }
                existing_student_ids = list(existing_student_options.keys())
                if st.session_state.get("cad_student_mode") == "existing" and not existing_student_ids:
                    st.session_state["cad_student_mode"] = "new"
                if st.session_state.get("cad_existing_student_id") not in existing_student_options:
                    st.session_state["cad_existing_student_id"] = existing_student_ids[0] if existing_student_ids else None

                st.radio(
                    "Como deseja identificar o aluno?",
                    options=["existing", "new"],
                    key="cad_student_mode",
                    horizontal=True,
                    format_func=lambda value: "Aluno já cadastrado" if value == "existing" else "Aluno novo",
                    disabled=not existing_student_ids,
                )

                previous_student_mode = st.session_state.get("cad_student_mode_previous")
                current_student_mode = st.session_state.get("cad_student_mode")
                if previous_student_mode != current_student_mode:
                    if current_student_mode == "new":
                        st.session_state["cad_nome"] = ""
                        st.session_state["cad_matricula"] = generated_matricula
                    elif current_student_mode == "existing" and existing_student_ids:
                        selected_existing_student = next(
                            (
                                row
                                for row in selected_class_students
                                if str(row["id"]) == str(st.session_state.get("cad_existing_student_id"))
                            ),
                            None,
                        )
                        if selected_existing_student:
                            st.session_state["cad_nome"] = _normalize_student_name(selected_existing_student.get("name", ""))
                            st.session_state["cad_matricula"] = _normalize_student_registration(
                                selected_existing_student.get("matricula", "")
                            )
                    st.session_state["cad_student_mode_previous"] = current_student_mode
                elif current_student_mode == "new" and not st.session_state.get("cad_matricula"):
                    st.session_state["cad_matricula"] = generated_matricula

                if selected_class_students_df.empty:
                    st.info("Nenhum aluno cadastrado nesta turma ainda. Cadastre um aluno novo para continuar.")

                selected_existing_student = None
                if st.session_state.get("cad_student_mode") == "existing" and existing_student_ids:
                    st.selectbox(
                        "Selecione o aluno",
                        options=existing_student_ids,
                        key="cad_existing_student_id",
                        format_func=lambda value: existing_student_options.get(value, ""),
                    )
                    selected_existing_student = next(
                        (
                            row
                            for row in selected_class_students
                            if str(row["id"]) == str(st.session_state.get("cad_existing_student_id"))
                        ),
                        None,
                    )
                    if selected_existing_student:
                        st.session_state["cad_nome"] = _normalize_student_name(selected_existing_student.get("name", ""))
                        st.session_state["cad_matricula"] = _normalize_student_registration(
                            selected_existing_student.get("matricula", "")
                        )

                form_col1, form_col2 = st.columns(2, gap="large")
                with form_col1:
                    nome_aluno = st.text_input(
                        "Nome do Aluno",
                        key="cad_nome",
                        placeholder="Digite o nome completo",
                        disabled=st.session_state.get("cad_student_mode") == "existing" and selected_existing_student is not None,
                    )
                with form_col2:
                    matricula = st.text_input(
                        "Matrícula",
                        key="cad_matricula",
                        placeholder="Gerada automaticamente",
                        disabled=(
                            st.session_state.get("cad_student_mode") == "new"
                            or (st.session_state.get("cad_student_mode") == "existing" and selected_existing_student is not None)
                        ),
                    )

                nome_norm = _normalize_student_name(nome_aluno)
                matr_norm = _normalize_student_registration(matricula)
                should_validate_identification = not (
                    st.session_state.get("cad_student_mode") == "existing" and selected_existing_student is not None
                )
                registration_errors = (
                    _validate_student_registration_fields(
                        nome_aluno,
                        matricula,
                        require_matricula=st.session_state.get("cad_student_mode") != "new",
                    )
                    if should_validate_identification and (
                        bool(nome_aluno)
                        if st.session_state.get("cad_student_mode") == "new"
                        else bool(nome_aluno or matricula)
                    )
                    else []
                )
                registration_ready = bool(nome_norm and matr_norm and not registration_errors and st.session_state.get("cad_class_id") is not None)

                if registration_errors:
                    for error in registration_errors:
                        st.warning(error)

                if not nome_norm:
                    st.markdown(
                        "<div style='border-radius:14px; padding:0.9rem 1rem; margin-top:0.9rem; "
                        "background:linear-gradient(180deg, rgba(72,50,148,0.32) 0%, rgba(48,32,95,0.28) 100%); "
                        "border:1px solid rgba(121,94,255,0.14); color:#C9C3FF;'>"
                        "<strong>Passo 1 de 2</strong><br/>Após preencher os dados, clique em <strong>Próximo</strong> para ir para a captura."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif not registration_ready:
                    st.info("Revise os campos informados para liberar a captura das imagens.")
                else:
                    if st.session_state.get("cad_student_mode") == "existing":
                        st.success("Aluno existente selecionado. A captura pode ser usada para atualizar as poses dele.")
                    else:
                        st.success("Dados validados. A captura já pode ser iniciada.")

                next_identification = st.button(
                    "➡️ Próximo para Captura",
                    disabled=not registration_ready,
                    use_container_width=True,
                )
                if next_identification:
                    locked_matricula_value = matricula
                    if st.session_state.get("cad_student_mode") == "new":
                        locked_matricula_value = get_next_student_registration()
                    st.session_state["cad_nome_locked"] = nome_aluno
                    st.session_state["cad_matricula_locked"] = locked_matricula_value
                    st.session_state["cad_class_id_locked"] = st.session_state.get("cad_class_id")
                    st.session_state["cad_student_mode_locked"] = st.session_state.get("cad_student_mode")
                    st.session_state["cad_existing_student_id_locked"] = st.session_state.get("cad_existing_student_id")
                    st.session_state["student_registration_stage"] = "capture"
                    st.rerun()
        else:
            selected_class_id = st.session_state.get("cad_class_id_locked", st.session_state.get("cad_class_id"))
            if locked_student_mode == "existing" and locked_existing_student_id:
                nome_criptografado = str(locked_existing_student_id)
            elif nome_norm and matr_norm:
                nome_criptografado = gerar_hash_nome_matricula(nome_norm, matr_norm)

            if nome_criptografado:
                os.makedirs(DATABASE_PATH, exist_ok=True)
                pasta_base = os.path.join(DATABASE_PATH, nome_criptografado)
                os.makedirs(pasta_base, exist_ok=True)
                for _pose in POSES:
                    os.makedirs(os.path.join(pasta_base, _pose), exist_ok=True)

            capture_left_col, capture_right_col = st.columns([0.95, 1.35], gap="large")

            with capture_left_col:
                with st.container(border=True):
                    _render_student_registration_card_header(
                        "📷",
                        "Controle da Captura",
                        "Acompanhe a pose atual e avance ao concluir cada etapa.",
                        "linear-gradient(180deg, #159957 0%, #0E7A46 100%)",
                    )
                    _render_student_registration_stepper(2)
                    st.markdown(
                        f"<div class='student-reg-info'>Aluno: <strong>{html.escape(nome_norm)}</strong><br/>"
                        f"Matrícula: <strong>{html.escape(matr_norm)}</strong><br/>"
                        f"Turma: <strong>{html.escape(class_options.get(selected_class_id, ''))}</strong></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("⬅️ Editar Identificação", use_container_width=True):
                        st.session_state["cad_nome"] = st.session_state.get("cad_nome_locked", st.session_state.get("cad_nome", ""))
                        st.session_state["cad_matricula"] = st.session_state.get("cad_matricula_locked", st.session_state.get("cad_matricula", ""))
                        if st.session_state.get("cad_class_id_locked") is not None:
                            st.session_state["cad_class_id"] = st.session_state.get("cad_class_id_locked")
                        st.session_state["student_registration_stage"] = "identify"
                        st.rerun()
                    st.markdown(
                        "<div class='student-reg-info'>"
                        "Ao concluir a ultima pose, o sistema finaliza o cadastro e registra o aluno automaticamente."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    progress_percent = int((img_index / max(IMAGENS_POR_POSE, 1)) * 100)
                    pose_label = CAPTURE_POSE_LABELS.get(pose_atual, pose_atual.replace("_", " ").title())
                    mini_col1, mini_col2 = st.columns(2, gap="large")
                    with mini_col1:
                        st.markdown(
                            f"<div class='student-reg-mini'><div class='student-reg-mini-label'>Pose atual</div>"
                            f"<div class='student-reg-mini-value'>{html.escape(pose_label)}</div>"
                            "<div class='student-reg-progress-track'><div class='student-reg-progress-fill' style='width:100%; "
                            "background:linear-gradient(90deg, #1B9E5A 0%, #39C875 100%);'></div></div></div>",
                            unsafe_allow_html=True,
                        )
                    with mini_col2:
                        st.markdown(
                            f"<div class='student-reg-mini'><div class='student-reg-mini-label'>Progresso</div>"
                            f"<div class='student-reg-mini-value'>{img_index} / {IMAGENS_POR_POSE} imagens</div>"
                            f"<div class='student-reg-progress-track'><div class='student-reg-progress-fill' style='width:{progress_percent}%;'></div></div></div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"<div class='student-reg-mini' style='margin-top:0.85rem;'><div class='student-reg-mini-label'>Etapa</div>"
                        f"<div class='student-reg-mini-value'>Etapa {pose_index + 1} de {len(POSES)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    _render_capture_pose_list(pose_index, POSES)

                    if pose_done:
                        st.markdown(
                            "<div class='student-reg-info' style='color:#83D9A1; border-color:rgba(39,194,110,0.16); "
                            "background:linear-gradient(180deg, rgba(18,85,54,0.32) 0%, rgba(15,58,39,0.22) 100%);'>"
                            "Pose concluída. Você já pode avançar para a próxima etapa."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    elif cap_running:
                        st.markdown(
                            "<div class='student-reg-info'>"
                            "Captura em andamento pelo navegador.<br/>"
                            "Use a webcam do seu notebook para tirar cada foto da pose atual. "
                            "Entre uma foto e outra, varie levemente o angulo, a distancia e a iluminacao."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='student-reg-info'>A webcam fica visível ao lado direito. Abra a câmera, ajuste a pose e então inicie a captura.</div>",
                            unsafe_allow_html=True,
                        )

                    button_col1, button_col2, button_col3 = st.columns(3)
                    with button_col1:
                        start_btn = st.button(
                            "▶️ Iniciar Captura",
                            disabled=cap_running or pose_done,
                            use_container_width=True,
                        )
                    with button_col2:
                        cancel_btn = st.button(
                            "⏹️ Cancelar",
                            disabled=not cap_running,
                            use_container_width=True,
                        )
                    with button_col3:
                        next_btn = st.button(
                            "➡️ Próximo",
                            disabled=st.session_state.get("cap_running", False) or not st.session_state.get("pose_done", False),
                            use_container_width=True,
                        )

                    st.markdown(
                        "<div class='student-reg-lock'>Inicie a captura quando a webcam estiver bem posicionada. Depois avance apenas quando a pose estiver concluída.</div>",
                        unsafe_allow_html=True,
                    )

            with capture_right_col:
                with st.container(border=True):
                    _render_student_registration_card_header(
                        "📷",
                        "Câmera de captura",
                        "A webcam do navegador sera usada para capturar as imagens desta pose.",
                        "linear-gradient(180deg, #5F49D6 0%, #4633A8 100%)",
                    )

        if registration_stage == "capture" and registration_ready and pasta_base:
            if start_btn:
                st.session_state.cap_running = True
                st.session_state.pose_done = False
                st.session_state.img_index = 0
                st.session_state.next_time = time.time() + prep_seconds
                cap_running = True
                img_index = 0
                next_time = st.session_state.next_time

            if cancel_btn:
                st.session_state.cap_running = False
                st.session_state.next_time = None
                cap_running = False

            if next_btn and pose_done:
                if (pose_index + 1) < len(POSES):
                    st.session_state.pose_index  = (pose_index + 1)
                    st.session_state.img_index   = 0
                    st.session_state.pose_done   = False
                    st.session_state.cap_running = False
                    st.session_state.next_time   = None
                    st.rerun()
                else:
                    persisted_student_id = _persist_completed_student_registration(
                        nome_criptografado,
                        nome_norm,
                        matr_norm,
                        selected_class_id,
                    )
                    st.session_state.registration_done = True
                    st.session_state.last_cad_nome = nome_norm
                    st.session_state.last_cad_matricula = matr_norm
                    st.session_state.last_cad_hash = persisted_student_id
                    st.session_state.last_embedding_status = None
                    st.session_state.last_embedding_message = ""
                    st.session_state.cap_running = False
                    st.session_state.pose_done   = False
                    st.session_state.next_time   = None
                    st.rerun()

            if registration_stage == "capture":
                with capture_right_col:
                    if webrtc_streamer is None or WebRtcMode is None or av is None:
                        st.error(
                            "A captura automática via WebRTC ainda não está disponível neste ambiente. "
                            "Reconstrua o app para instalar `streamlit-webrtc`."
                        )
                        if WEBRTC_IMPORT_ERROR is not None:
                            st.caption(f"Detalhe técnico: {WEBRTC_IMPORT_ERROR}")
                        webrtc_ctx = None
                        processor = None
                        camera_ready = False
                    else:
                        webrtc_ctx = webrtc_streamer(
                            key="cadastro_webrtc_stream",
                            mode=WebRtcMode.SENDRECV,
                            rtc_configuration={
                                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
                            },
                            media_stream_constraints={
                                "video": {
                                    "width": {"ideal": 1280},
                                    "height": {"ideal": 720},
                                    "facingMode": "user",
                                },
                                "audio": False,
                            },
                            video_processor_factory=StudentRegistrationVideoProcessor,
                            async_processing=True,
                        )
                        processor = webrtc_ctx.video_processor if webrtc_ctx else None
                        camera_ready = bool(webrtc_ctx and webrtc_ctx.state.playing and processor is not None)

                    if start_btn and not camera_ready:
                        st.session_state.cap_running = False
                        st.session_state.next_time = None
                        cap_running = False
                        st.warning("Abra a webcam pelo controle START acima antes de iniciar a captura automática.")

                    if cap_running:
                        pasta_pose = os.path.join(pasta_base, pose_atual)
                        os.makedirs(pasta_pose, exist_ok=True)
                        if not camera_ready:
                            st.session_state.cap_running = False
                            st.session_state.next_time = None
                            st.warning("A webcam foi interrompida. Clique em START acima para reabrir o vídeo e depois inicie a captura novamente.")
                        else:
                            frame_bgr = processor.get_latest_frame_bgr()
                            now = time.time()
                            restante = max(0.0, (st.session_state.next_time or now) - now)

                            if frame_bgr is None:
                                st.info("Aguardando os primeiros frames da webcam...")
                            elif restante > 0.0:
                                st.info(f"Prepare a pose. A próxima captura automática será feita em {restante:0.1f}s.")
                            else:
                                timestamp = get_local_now().strftime("%Y%m%d_%H%M%S%f")
                                nome_arquivo = f"{pose_atual}_{timestamp}.jpg"
                                caminho = os.path.join(pasta_pose, nome_arquivo)
                                cv2.imwrite(caminho, frame_bgr)
                                st.session_state.img_index += 1
                                st.session_state.next_time = now + capture_interval
                                st.success(f"Imagem {st.session_state.img_index} de {IMAGENS_POR_POSE} capturada automaticamente.")

                                if st.session_state.img_index >= IMAGENS_POR_POSE:
                                    st.session_state.cap_running = False
                                    st.session_state.pose_done = True
                                    st.session_state.next_time = None

                                    if pose_index == len(POSES) - 1:
                                        persisted_student_id = _persist_completed_student_registration(
                                            nome_criptografado,
                                            nome_norm,
                                            matr_norm,
                                            selected_class_id,
                                        )
                                        st.session_state.registration_done = True
                                        st.session_state.last_cad_nome = nome_norm
                                        st.session_state.last_cad_matricula = matr_norm
                                        st.session_state.last_cad_hash = persisted_student_id
                                        st.session_state.last_embedding_status = None
                                        st.session_state.last_embedding_message = ""
                                    else:
                                        st.success(
                                            f"✅ {IMAGENS_POR_POSE} imagens capturadas para '{pose_atual}'. "
                                            f"Clique em **Próximo** para a próxima pose."
                                        )
                                    st.rerun()

                            if st.session_state.get("cap_running", False):
                                time.sleep(0.18)
                                st.rerun()
                    else:
                        if pose_done:
                            st.success("Pose concluida. Clique em **Próximo** para seguir para a proxima etapa.")
                        elif camera_ready:
                            st.info("Webcam pronta. Clique em **Iniciar Captura** para começar a captura automática por intervalo.")
                        else:
                            st.info("Clique em START acima para abrir a webcam do notebook. Depois use **Iniciar Captura** para começar a coleta automática.")
    # ------------------ MONITORAMENTO ------------------
    elif menu_option == "Monitoramento":
        _release_registration_camera()

        st.markdown(
            """
            <style>
            .monitor-page-shell {
                margin-top: 0.2rem;
            }
            .monitor-page-hero {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin: 0.15rem 0 1.35rem 0;
            }
            .monitor-page-hero-icon {
                width: 64px;
                height: 64px;
                border-radius: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, rgba(104,84,255,0.24) 0%, rgba(63,50,145,0.18) 100%);
                border: 1px solid rgba(126,107,255,0.24);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            }
            .monitor-page-hero-icon svg {
                width: 40px;
                height: 40px;
                display: block;
            }
            .monitor-page-title {
                margin: 0;
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 800;
                letter-spacing: 0.01em;
            }
            .monitor-page-subtitle {
                margin: 0.32rem 0 0 0;
                color: #A7B0C0;
                font-size: 1rem;
                line-height: 1.5;
            }
            .monitor-layout {
                margin-top: 0;
            }
            .monitor-layout [data-testid="column"] > div {
                height: 100%;
            }
            .monitor-card {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(21,25,36,0.94) 0%, rgba(17,21,31,0.98) 100%);
                box-shadow: 0 16px 34px rgba(0,0,0,0.16);
                padding: 1.2rem 1.2rem 1.1rem 1.2rem;
                margin-bottom: 1rem;
            }
            .monitor-card-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.9rem;
                margin-bottom: 1rem;
            }
            .monitor-card-title-wrap {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                min-width: 0;
            }
            .monitor-card-icon {
                width: 40px;
                height: 40px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
                color: #F5F7FB;
                flex: 0 0 auto;
            }
            .monitor-card-icon svg {
                width: 22px;
                height: 22px;
                display: block;
            }
            .monitor-card-icon-purple {
                background: linear-gradient(180deg, #6E58FF 0%, #4A36C9 100%);
            }
            .monitor-card-icon-neutral {
                background: linear-gradient(180deg, rgba(94,102,121,0.3) 0%, rgba(56,63,79,0.32) 100%);
                color: #B8C1D3;
            }
            .monitor-card-icon-green {
                background: linear-gradient(180deg, #1F9B61 0%, #147446 100%);
            }
            .monitor-card-title {
                font-size: 1.06rem;
                font-weight: 800;
                color: #F4F7FB;
                margin: 0;
            }
            .monitor-field-gap {
                margin-top: 0.35rem;
            }
            .monitor-context-card {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(21,25,36,0.94) 0%, rgba(17,21,31,0.98) 100%);
                box-shadow: 0 16px 34px rgba(0,0,0,0.16);
                padding: 1.2rem;
            }
            .monitor-context-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.9rem 1rem;
            }
            .monitor-context-item {
                border-top: 1px solid rgba(255,255,255,0.06);
                padding-top: 0.85rem;
            }
            .monitor-context-label {
                color: #8F9AAF;
                font-size: 0.82rem;
                margin-bottom: 0.2rem;
            }
            .monitor-context-value {
                color: #F5F7FB;
                font-size: 1rem;
                font-weight: 700;
            }
            .monitor-tip-box {
                border-radius: 16px;
                padding: 0.95rem 1rem;
                background: linear-gradient(180deg, rgba(55,34,111,0.42) 0%, rgba(34,22,66,0.34) 100%);
                border: 1px solid rgba(124,95,255,0.14);
                color: #D4D9E6;
                line-height: 1.55;
            }
            .monitor-tip-title {
                font-size: 1rem;
                font-weight: 800;
                color: #F4F7FB;
                margin-bottom: 0.28rem;
            }
            .monitor-video-card {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px;
                background: linear-gradient(180deg, rgba(21,25,36,0.94) 0%, rgba(17,21,31,0.98) 100%);
                box-shadow: 0 16px 34px rgba(0,0,0,0.18);
                overflow: hidden;
            }
            .monitor-video-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 1.15rem 1.2rem;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            .monitor-video-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.42rem 0.9rem;
                border-radius: 999px;
                font-size: 0.86rem;
                font-weight: 700;
            }
            .monitor-video-badge.waiting {
                color: #49D17F;
                background: rgba(28,132,75,0.16);
                border: 1px solid rgba(46,176,101,0.26);
            }
            .monitor-video-badge.live {
                color: #49D17F;
                background: rgba(28,132,75,0.16);
                border: 1px solid rgba(46,176,101,0.26);
            }
            .monitor-video-badge.ended {
                color: #FFCC73;
                background: rgba(168,112,22,0.16);
                border: 1px solid rgba(226,163,65,0.24);
            }
            .monitor-video-metrics {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                padding: 1rem 1.2rem 0.25rem 1.2rem;
            }
            .monitor-metric-box {
                border-radius: 16px;
                background: rgba(255,255,255,0.025);
                border: 1px solid rgba(255,255,255,0.05);
                padding: 0.85rem 0.95rem;
            }
            .monitor-metric-label {
                color: #95A0B4;
                font-size: 0.86rem;
                margin-bottom: 0.18rem;
            }
            .monitor-metric-value {
                color: #F5F7FB;
                font-size: 1.05rem;
                font-weight: 800;
            }
            .monitor-video-stage {
                padding: 1rem 1.2rem 1.2rem 1.2rem;
            }
            .monitor-video-shell {
                min-height: 430px;
                border-radius: 20px;
                border: 2px dashed rgba(124,95,255,0.46);
                background: linear-gradient(180deg, rgba(14,18,28,0.9) 0%, rgba(17,21,31,0.96) 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .monitor-placeholder {
                min-height: 430px;
                border-radius: 20px;
                border: 2px dashed rgba(124,95,255,0.46);
                background: linear-gradient(180deg, rgba(14,18,28,0.9) 0%, rgba(17,21,31,0.96) 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 2rem 1.4rem;
            }
            .monitor-placeholder-icon {
                width: 88px;
                height: 88px;
                margin: 0 auto 1.2rem auto;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                background: linear-gradient(180deg, rgba(106,82,255,0.28) 0%, rgba(74,56,175,0.24) 100%);
                color: #8E7BFF;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            }
            .monitor-placeholder-icon svg {
                width: 42px;
                height: 42px;
                display: block;
            }
            .monitor-placeholder-title {
                color: #F4F7FB;
                font-size: 1.75rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }
            .monitor-placeholder-subtitle {
                color: #A8B1C0;
                font-size: 1rem;
                line-height: 1.55;
            }
            .monitor-info-bar {
                margin: 0 1.2rem 1.2rem 1.2rem;
                border-radius: 16px;
                background: linear-gradient(180deg, rgba(26,57,104,0.46) 0%, rgba(20,44,84,0.36) 100%);
                border: 1px solid rgba(68,126,214,0.18);
                color: #8FC0FF;
                padding: 0.95rem 1rem;
                line-height: 1.5;
            }
            .monitor-focus-shell {
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px;
                background: linear-gradient(180deg, rgba(21,25,36,0.94) 0%, rgba(17,21,31,0.98) 100%);
                box-shadow: 0 16px 34px rgba(0,0,0,0.18);
                padding: 1.15rem 1.2rem;
                margin-bottom: 1rem;
            }
            .monitor-focus-kicker {
                color: #8FC0FF;
                font-size: 0.8rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.38rem;
            }
            .monitor-focus-title {
                color: #F5F7FB;
                font-size: 1.3rem;
                font-weight: 800;
                margin: 0 0 0.25rem 0;
            }
            .monitor-focus-subtitle {
                color: #A8B1C0;
                font-size: 0.95rem;
                line-height: 1.5;
                margin: 0;
            }
            @media (max-width: 1100px) {
                .monitor-page-hero {
                    align-items: flex-start;
                }
                .monitor-video-metrics,
                .monitor-context-grid {
                    grid-template-columns: 1fr;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        CONFIDENCE_THRESHOLD = 0.35
        show_debug = False
        show_unknown_boxes = False
        debug_font = 0.8
        device = "cuda" if torch.cuda.is_available() else "cpu"
        BOX_MARGIN_RATIO = 0.2
        if user_role != "professor":
            st.info("O monitoramento operacional está disponível apenas para o perfil professor.")
            return

        subjects_df = list_subjects_for_user(user_context)
        if subjects_df.empty:
            st.warning("Nenhuma disciplina vinculada ao professor foi encontrada. Cadastre os vínculos no banco antes de iniciar o monitoramento.")
            return

        monitor_state = st.session_state.setdefault("monitoring_state", {})
        current_session_id = monitor_state.get("session_id")
        current_session_preview = get_monitoring_session_summary(current_session_id) if current_session_id else None
        header_is_live = bool(current_session_preview and current_session_preview["status"] == SESSION_STATUS_OPEN)
        selected_subject_id = monitor_state.get("selected_subject_id")
        selected_class_id = monitor_state.get("selected_class_id")
        selected_lesson_type = monitor_state.get("selected_lesson_type", DEFAULT_LESSON_TYPE)
        video_snapshot = _get_monitor_video_snapshot(current_session_preview)
        subject_widget_key = "monitor_selected_subject_label"
        class_widget_key = "monitor_selected_class_label"
        lesson_type_widget_key = "monitor_selected_lesson_type"

        st.markdown(
            """
            <div class="monitor-page-shell">
                <div class="monitor-page-hero">
                    <div class="monitor-page-hero-icon" aria-hidden="true">
                        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M33 10C24.2 10 17 17.2 17 26V30.2C17 33.4 15.8 36.5 13.7 38.9L11.8 41.1C10.3 42.8 11.5 45.5 13.8 45.5H52.2C54.5 45.5 55.7 42.8 54.2 41.1L52.3 38.9C50.2 36.5 49 33.4 49 30.2V26C49 17.2 41.8 10 33 10Z" fill="#E9EEF9" fill-opacity="0.96"/>
                            <path d="M24.5 45.5C25.5 50.2 29 53 33 53C37 53 40.5 50.2 41.5 45.5H24.5Z" fill="#DCE4F4"/>
                            <path d="M23 22.5C25.8 18.2 30.2 15.8 35.2 15.8C37.4 15.8 39.6 16.3 41.5 17.2" stroke="#B9C5DA" stroke-width="3" stroke-linecap="round"/>
                            <path d="M16.2 21.8C18 18.7 20.7 16.1 23.9 14.4" stroke="#8AAAF6" stroke-width="3.2" stroke-linecap="round"/>
                            <circle cx="45.5" cy="17.5" r="2.8" fill="#6E58FF"/>
                            <path d="M29 48.6C30.1 50.3 31.4 51.1 33 51.1C34.6 51.1 35.9 50.3 37 48.6" stroke="#B7C3D9" stroke-width="2.6" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="monitor-page-title">Monitoramento em Tempo Real</h1>
                        <p class="monitor-page-subtitle">Acompanhe a sessão da turma selecionada.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_closed_session = st.session_state.get("last_closed_monitoring_session")
        run_system = False
        stop_system = False

        subject_options = {int(row["id"]): row["nome"] for _, row in subjects_df.iterrows()}
        default_subject_index = 0
        if selected_subject_id in subject_options:
            default_subject_index = list(subject_options.keys()).index(selected_subject_id)
        disable_scope_inputs = bool(current_session_preview and current_session_preview["status"] == SESSION_STATUS_OPEN)
        default_subject_label = list(subject_options.values())[default_subject_index]
        if st.session_state.get(subject_widget_key) not in subject_options.values():
            st.session_state[subject_widget_key] = default_subject_label

        if disable_scope_inputs:
            selected_subject_label = subject_options.get(selected_subject_id, default_subject_label)
        else:
            selected_subject_label = default_subject_label

        selected_subject_id = next(key for key, value in subject_options.items() if value == selected_subject_label)
        monitor_state["selected_subject_id"] = selected_subject_id
        st.session_state["monitoring_state"] = monitor_state

        classes_df = list_classes_for_user(user_context, selected_subject_id)
        if classes_df.empty:
            st.warning("Nenhuma turma vinculada à disciplina selecionada foi encontrada para este professor.")
            return

        class_options = {
            int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
            for _, row in classes_df.iterrows()
        }
        default_class_index = 0
        if selected_class_id in class_options:
            default_class_index = list(class_options.keys()).index(selected_class_id)
        default_class_label = list(class_options.values())[default_class_index]
        if st.session_state.get(class_widget_key) not in class_options.values():
            st.session_state[class_widget_key] = default_class_label

        if disable_scope_inputs:
            selected_class_label = class_options.get(selected_class_id, default_class_label)
        else:
            selected_class_label = default_class_label

        selected_class_id = next(key for key, value in class_options.items() if value == selected_class_label)
        monitor_state["selected_class_id"] = selected_class_id
        st.session_state["monitoring_state"] = monitor_state

        if selected_lesson_type not in LESSON_TYPE_OPTIONS:
            selected_lesson_type = DEFAULT_LESSON_TYPE
        if st.session_state.get(lesson_type_widget_key) not in LESSON_TYPE_OPTIONS:
            st.session_state[lesson_type_widget_key] = selected_lesson_type

        if not professor_has_assignment(user_context["teacher_id"], selected_subject_id, selected_class_id):
            st.error("O professor autenticado não possui vínculo com a disciplina e a turma selecionadas.")
            return

        current_session_id = monitor_state.get("session_id")
        current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None
        if current_session and current_session["status"] == SESSION_STATUS_OPEN:
            ui_state = "Em andamento"
        elif last_closed_session:
            ui_state = "Encerrado"
        else:
            ui_state = "Não iniciado"

        if ui_state == "Em andamento":
            session_panel = current_session
        elif ui_state == "Encerrado":
            session_panel = last_closed_session
        else:
            session_panel = None

        is_focus_mode = ui_state == "Em andamento"

        if not is_focus_mode:
            monitor_left_col, monitor_right_col = st.columns([1.02, 2.05], gap="large")
            with monitor_left_col:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="monitor-card-head" style="margin-bottom:0.85rem;">
                            <div class="monitor-card-title-wrap">
                                <div class="monitor-card-icon monitor-card-icon-neutral">☷</div>
                                <div class="monitor-card-title">Seleção da Sessão</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    selected_subject_label = st.selectbox(
                        "Disciplina da aula",
                        list(subject_options.values()),
                        key=subject_widget_key,
                        disabled=disable_scope_inputs,
                    )
                    selected_subject_id = next(key for key, value in subject_options.items() if value == selected_subject_label)
                    monitor_state["selected_subject_id"] = selected_subject_id
                    st.session_state["monitoring_state"] = monitor_state

                    classes_df = list_classes_for_user(user_context, selected_subject_id)
                    if classes_df.empty:
                        st.warning("Nenhuma turma vinculada à disciplina selecionada foi encontrada para este professor.")
                        return

                    class_options = {
                        int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
                        for _, row in classes_df.iterrows()
                    }
                    default_class_index = 0
                    if selected_class_id in class_options:
                        default_class_index = list(class_options.keys()).index(selected_class_id)
                    default_class_label = list(class_options.values())[default_class_index]
                    if st.session_state.get(class_widget_key) not in class_options.values():
                        st.session_state[class_widget_key] = default_class_label

                    selected_class_label = st.selectbox(
                        "Turma acompanhada",
                        list(class_options.values()),
                        key=class_widget_key,
                        disabled=disable_scope_inputs,
                    )
                    selected_class_id = next(key for key, value in class_options.items() if value == selected_class_label)
                    monitor_state["selected_class_id"] = selected_class_id
                    st.session_state["monitoring_state"] = monitor_state

                    selected_lesson_type = st.selectbox(
                        "Tipo de aula",
                        LESSON_TYPE_OPTIONS,
                        key=lesson_type_widget_key,
                        disabled=disable_scope_inputs,
                    )
                    monitor_state["selected_lesson_type"] = selected_lesson_type
                    st.session_state["monitoring_state"] = monitor_state

                _render_context_card(ui_state, selected_subject_label, selected_class_label, session_panel)

                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="monitor-card-head" style="margin-bottom:0.85rem;">
                            <div class="monitor-card-title-wrap">
                                <div class="monitor-card-icon monitor-card-icon-neutral">⎋</div>
                                <div class="monitor-card-title">Ações</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if ui_state == "Não iniciado":
                        run_system = st.button("▶ Iniciar Monitoramento", type="primary", use_container_width=True)
                        st.button("■ Finalizar Sessão", disabled=True, use_container_width=True)
                    else:
                        run_system = st.button("▶ Iniciar nova sessão", type="primary", use_container_width=True)
                        st.button("■ Finalizar Sessão", disabled=True, use_container_width=True)

                    st.markdown(
                        """
                        <div class="monitor-tip-box" style="margin-top:1rem;">
                            <div class="monitor-tip-title">Como funciona?</div>
                            Ao iniciar, o sistema começará a capturar e analisar os comportamentos automaticamente.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            monitor_right_col = st.container()
            with st.container():
                focus_info_col, focus_action_col = st.columns([3.2, 1.1], gap="large")
                with focus_info_col:
                    st.markdown(
                        f"""
                        <div class="monitor-focus-shell">
                            <div class="monitor-focus-kicker">Sessão em andamento</div>
                            <div class="monitor-focus-title">Vídeo em modo ampliado</div>
                            <p class="monitor-focus-subtitle">
                                {html.escape(selected_subject_label)} • {html.escape(selected_class_label)} • {html.escape(selected_lesson_type)}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with focus_action_col:
                    stop_system = st.button("■ Encerrar Monitoramento", type="primary", use_container_width=True)

        current_session_id = monitor_state.get("session_id")
        current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None

        if run_system:
            session_id = create_monitoring_session(
                user_context["teacher_id"],
                selected_subject_id,
                selected_class_id,
                selected_lesson_type,
            )
            session_summary = get_monitoring_session_summary(session_id)
            st.session_state["student_lookup"] = get_student_lookup_for_scope(user_context)
            st.session_state["monitoring_state"] = {
                "session_id": session_id,
                "selected_subject_id": selected_subject_id,
                "selected_class_id": selected_class_id,
                "selected_lesson_type": selected_lesson_type,
            }
            st.session_state.pop("last_closed_monitoring_session", None)
            st.session_state["episode_manager"] = BehaviorEpisodeManager(
                persist_callback=lambda **kwargs: insert_behavior_episode(
                    monitoring_session_id=session_id,
                    student_id=kwargs.get("id_student"),
                    lesson_type=session_summary.get("lesson_type"),
                    school=school,
                    discipline=session_summary["subject_name"],
                    teacher=user_name,
                    id_student=kwargs.get("id_student"),
                    student=kwargs.get("student"),
                    behavior=kwargs.get("behavior"),
                    start_time=kwargs.get("start_time"),
                    end_time=kwargs.get("end_time"),
                    source=kwargs.get("source", "realtime"),
                ),
                stability_seconds=2.0,
                stability_frames=15,
            )
            st.session_state["monitor_runtime"] = build_monitor_runtime(device, RELAY_HOST, RELAY_PORT)
            st.session_state.pop("monitor_waiting_since", None)
            st.session_state.pop("monitor_last_display_frame", None)
            st.session_state.pop("monitor_last_detector_frame_id", None)
            st.session_state.pop("monitor_last_rendered_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_at", None)
            st.session_state.pop("monitor_last_overlays", None)
            st.session_state.pop("monitor_last_frame_shape", None)
            st.rerun()

        monitor_state = st.session_state.get("monitoring_state", {})
        current_session_id = monitor_state.get("session_id")
        current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None

        if current_session and "monitor_runtime" not in st.session_state:
            st.session_state["student_lookup"] = get_student_lookup_for_scope(user_context)
            if "episode_manager" not in st.session_state:
                st.session_state["episode_manager"] = BehaviorEpisodeManager(
                    persist_callback=lambda **kwargs: insert_behavior_episode(
                        monitoring_session_id=current_session["id"],
                        student_id=kwargs.get("id_student"),
                        lesson_type=current_session.get("lesson_type"),
                        school=school,
                        discipline=current_session["subject_name"],
                        teacher=user_name,
                        id_student=kwargs.get("id_student"),
                        student=kwargs.get("student"),
                        behavior=kwargs.get("behavior"),
                        start_time=kwargs.get("start_time"),
                        end_time=kwargs.get("end_time"),
                        source=kwargs.get("source", "realtime"),
                    ),
                    stability_seconds=2.0,
                    stability_frames=15,
                )
            with st.spinner("Preparando runtime de monitoramento..."):
                st.session_state["monitor_runtime"] = build_monitor_runtime(device, RELAY_HOST, RELAY_PORT)
            st.session_state.pop("monitor_waiting_since", None)
            st.session_state.pop("monitor_last_display_frame", None)
            st.session_state.pop("monitor_last_detector_frame_id", None)
            st.session_state.pop("monitor_last_rendered_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_at", None)
            st.session_state.pop("monitor_last_overlays", None)
            st.session_state.pop("monitor_last_frame_shape", None)
            st.rerun()

        with monitor_right_col:
            with st.container(border=True):
                monitor_summary_placeholder = st.empty()
                render_monitor_video_summary_fragment(
                    current_session_id=current_session_id,
                    placeholder=monitor_summary_placeholder,
                )
                if current_session:
                    with st.container(border=True):
                        monitor_status_placeholder = st.empty()
                        monitor_frame_placeholder = st.empty()
                        process_monitor_fragment(
                            school=school,
                            discipline=current_session["subject_name"],
                            user_name=user_name,
                            current_session_id=current_session_id,
                            confidence_threshold=CONFIDENCE_THRESHOLD,
                            show_debug=show_debug,
                            debug_font=debug_font,
                            box_margin_ratio=BOX_MARGIN_RATIO,
                            show_unknown_boxes=show_unknown_boxes,
                            summary_placeholder=monitor_summary_placeholder,
                            status_placeholder=monitor_status_placeholder,
                            frame_placeholder=monitor_frame_placeholder,
                        )
                else:
                    st.markdown(
                        f"""
                        <div class="monitor-placeholder">
                            <div>
                                <div class="monitor-placeholder-icon">{_monitor_hero_icon_svg()}</div>
                                <div class="monitor-placeholder-title">Tela de Monitoramento</div>
                                <div class="monitor-placeholder-subtitle">
                                    O vídeo da câmera será exibido aqui após o início da sessão.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    """
                    <div class="monitor-info-bar" style="margin:1rem 0 0 0;">
                        A captura e a análise comportamental serão iniciadas automaticamente.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if stop_system and current_session_id:
            episode_manager = st.session_state.get("episode_manager")
            if episode_manager is not None and current_session is not None:
                episode_manager.flush_all(
                    timestamp=get_local_now(),
                    school=school,
                    discipline=current_session["subject_name"],
                    teacher=user_name,
                    source="realtime",
                )
            close_monitoring_session(current_session_id, status=SESSION_STATUS_CLOSED)
            closed_session = get_monitoring_session_summary(current_session_id)
            teardown_monitor_runtime()
            st.session_state.pop("episode_manager", None)
            st.session_state.pop("student_lookup", None)
            st.session_state.pop("monitoring_state", None)
            st.session_state.pop(subject_widget_key, None)
            st.session_state.pop(class_widget_key, None)
            st.session_state.pop(lesson_type_widget_key, None)
            st.session_state["last_closed_monitoring_session"] = closed_session
            st.session_state.pop("monitor_waiting_since", None)
            st.session_state.pop("monitor_last_display_frame", None)
            st.session_state.pop("monitor_last_detector_frame_id", None)
            st.session_state.pop("monitor_last_rendered_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_frame_id", None)
            st.session_state.pop("monitor_last_sync_fallback_at", None)
            st.session_state.pop("monitor_last_overlays", None)
            st.session_state.pop("monitor_last_frame_shape", None)
            st.success("Sessão de monitoramento encerrada com sucesso.")
            st.rerun()

    # ------------------ GRÁFICOS ------------------
    elif menu_option == "Usuários":
        render_admin_user_page(user_context)

    # ------------------ GRÁFICOS ------------------
    elif menu_option == "Gráficos":
        show_behavior_charts(user_context)

    # ------------------ RELATÓRIOS ------------------
    elif menu_option == "Relatórios":
        render_report_page(user_context)
