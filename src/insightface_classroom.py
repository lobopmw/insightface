
import base64
import html
import os
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
from control_database_postgres import (
    APP_TIMEZONE,
    DEFAULT_SCHOOL_NAME,
    SESSION_STATUS_OPEN,
    SESSION_STATUS_CLOSED,
    close_monitoring_session,
    create_monitoring_session,
    get_local_now,
    get_monitoring_session_summary,
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
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from insightface.app import FaceAnalysis
import warnings
import hashlib
from utils_criptografia import salvar_mapeamento
from socket_video_stream import VideoStream  # cliente do relay via socket
import threading
from collections import deque
from behavior_episode_service import BehaviorEpisodeManager
from ui.admin_user_page import render_admin_user_page
from ui.report_page import render_report_page

warnings.filterwarnings("ignore", category=FutureWarning)

# Suavização de "Dormindo"
sleep_smoother = {}
ENTER_SLEEP_FRAMES = 6  
EXIT_SLEEP_FRAMES  = 10  

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

lateral_timers = {}
DISTRACTED_TIMEOUT_SECONDS = 2.5
UNKNOWN_IDENTITY_LABELS = {"desconhecido", "unknown", ""}

FACE_DET_SIZE_GPU = (1280, 1280)
FACE_DET_SIZE_CPU = (960, 960)
POSE_IMGSZ_GPU = 1280
POSE_IMGSZ_CPU = 960
POSE_DET_CONF = 0.22
FACE_RECOGNITION_BASE_THRESHOLD = 0.45
FACE_RECOGNITION_MEDIUM_THRESHOLD = 0.41
FACE_RECOGNITION_SMALL_THRESHOLD = 0.37
FACE_RECOGNITION_MIN_MARGIN = 0.015

CAPTURE_POSE_LABELS = {
    "frontal": "Frontal",
    "lateral_esquerda": "Esquerda",
    "lateral_direita": "Direita",
    "cabeca_baixa": "Cabeça baixa",
}


def _normalize_student_name(value: str) -> str:
    return " ".join((value or "").strip().split())


def _normalize_student_registration(value: str) -> str:
    return "".join((value or "").strip().split())


def _validate_student_registration_fields(name: str, matricula: str) -> list[str]:
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

    if not normalized_matricula:
        errors.append("Informe a matrícula do aluno.")
    elif not normalized_matricula.isdigit():
        errors.append("O campo matrícula deve conter apenas números.")
    elif len(normalized_matricula) < 3:
        errors.append("A matrícula deve ter pelo menos 3 dígitos.")

    return errors


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
        """
        <div class="student-reg-header">
            <div class="student-reg-icon">📸</div>
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
    st.markdown(
        """
        <style>
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
            min-height: 54px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(180deg, rgba(30,33,52,0.94) 0%, rgba(21,24,38,0.96) 100%);
            color: #FFFFFF;
            font-size: 1.35rem;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(255,255,255,0.22);
            color: #FFFFFF;
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
        """,
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


@st.cache_resource(show_spinner=False)
def build_monitor_runtime(device: str, relay_host: str, relay_port: int):
    model = YOLO('yolo11m-pose.pt')

    if device == "cuda":
        model_face = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        model_face.prepare(ctx_id=0, det_size=FACE_DET_SIZE_GPU)
    else:
        model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        model_face.prepare(ctx_id=-1, det_size=FACE_DET_SIZE_CPU)

    known_face_encodings, known_face_names = load_insightface_data()
    known_face_encodings_norm = (
        known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
    ) if len(known_face_encodings) > 0 else None

    video_stream = VideoStream((relay_host, relay_port)).start()
    pose_imgsz = POSE_IMGSZ_GPU if device == "cuda" else POSE_IMGSZ_CPU
    detector = DetectorWorker(
        model,
        model_face,
        device,
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


@st.fragment(run_every=1.0)
def render_monitor_video_summary_fragment(current_session_id=None):
    current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None
    video_snapshot = _get_monitor_video_snapshot(current_session)
    st.markdown(
        f"""
        <div class="monitor-video-head" style="padding:0 0 1rem 0; border-bottom:none;">
            <div class="monitor-card-title-wrap">
                <div class="monitor-card-icon monitor-card-icon-purple">📷</div>
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
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1.0)
def render_context_fragment(session_state_label: str, selected_subject_label: str, selected_class_label: str, session_data=None):
    _render_context_card(session_state_label, selected_subject_label, selected_class_label, session_data)


@st.fragment(run_every=0.6)
def process_monitor_fragment(
    school: str,
    discipline: str,
    user_name: str,
    confidence_threshold: float,
    show_debug: bool,
    debug_font: float,
    box_margin_ratio: float,
    show_unknown_boxes: bool,
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
    frame_id = 0
    st.session_state["monitor_live_stats"] = {
        "detected_faces_count": 0,
        "recognized_faces_count": 0,
        "rendered_tracks_count": 0,
    }
    if video_stream is not None:
        if hasattr(video_stream, "read_with_meta"):
            frame, frame_id, _ = video_stream.read_with_meta()
        else:
            frame = video_stream.read()
            status = video_stream.get_status() if hasattr(video_stream, "get_status") else {}
            frame_id = status.get("frame_id", status.get("frames_received", 0))
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

        status_placeholder.warning(
            "\n".join(
                [
                    f"Aguardando frames do relay `{status['server'][0]}:{status['server'][1]}`",
                    f"Conectado: {'sim' if status['connected'] else 'nao'}",
                    f"Frames recebidos: {status['frames_received']}",
                    f"Ultimo frame ha: {f'{last_frame_age:.1f}s' if last_frame_age is not None else 'nenhum'}",
                    f"Ultimo erro: {status['last_error'] or 'nenhum'}",
                ]
            )
        )
        if waited > 10:
            status_placeholder.error(
                "O app conectou no relay, mas nao recebeu frame util a tempo. "
                "Valide os logs do container `relay` e a estabilidade do RTSP."
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
    results, faces = detector.get_outputs()

    face_named = []
    detected_faces_count = 0
    recognized_faces_count = 0
    if faces:
        detected_faces_count = len(faces)
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            name_face = identify_face(face, known_face_encodings_norm, known_face_names)
            face_named.append(((fx1, fy1, fx2, fy2), name_face))
            if name_face != "Desconhecido":
                recognized_faces_count += 1
                remember_name((fx1, fy1, fx2, fy2), name_face)

    rendered_tracks_count = 0
    hidden_unknown_tracks_count = 0

    if results:
        for result in results:
            if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
                continue
            keypoints_all = result.keypoints.data.cpu().numpy()
            hud_lines = []

            for pid, person_keypoints in enumerate(keypoints_all):
                if len(person_keypoints) == 0:
                    continue

                current_behavior = "Atento"
                have_all = False

                if person_keypoints.shape[0] > 10:
                    nose = person_keypoints[0]
                    ls, rs = person_keypoints[5], person_keypoints[6]
                    le, re = person_keypoints[7], person_keypoints[8]
                    lw, rw = person_keypoints[9], person_keypoints[10]

                    confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
                    have_all = all(c > confidence_threshold for c in confs)
                    if have_all:
                        current_behavior = classify_behavior(nose, ls, rs, le, re, lw, rw, confidence_threshold)

                x_coords = [p[0] for p in person_keypoints if p[2] > confidence_threshold]
                y_coords = [p[1] for p in person_keypoints if p[2] > confidence_threshold]
                if not x_coords or not y_coords:
                    continue
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                y_min = max(0, int(y_min - box_margin_ratio * (y_max - y_min)))
                person_box = (x_min, y_min, x_max, y_max)

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

                    if nose[2] > confidence_threshold or (ls[2] > confidence_threshold and rs[2] > confidence_threshold):
                        lateral_status = is_lateral_view(
                            nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=confidence_threshold
                        )
                        back_status = is_back_view(
                            nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=confidence_threshold
                        )
                        new_behavior = check_distracted_status(
                            behavior_key, (lateral_status or back_status), lateral_timers, timeout=DISTRACTED_TIMEOUT_SECONDS
                        )
                        if new_behavior:
                            current_behavior = new_behavior

                    raw_behavior = current_behavior
                    state = sleep_smoother.setdefault(behavior_key, {"state":"Atento","sleep":0,"awake":0})

                    if raw_behavior == "Dormindo":
                        state["sleep"] += 1
                        state["awake"] = 0
                        if state["state"] != "Dormindo" and state["sleep"] >= ENTER_SLEEP_FRAMES:
                            state["state"] = "Dormindo"
                    else:
                        state["awake"] += 1
                        state["sleep"] = 0
                        if state["state"] == "Dormindo" and state["awake"] >= EXIT_SLEEP_FRAMES:
                            state["state"] = raw_behavior
                        elif state["state"] != "Dormindo":
                            state["state"] = raw_behavior

                    current_behavior = state["state"]

                if name_student != "Desconhecido" and episode_manager is not None:
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

                box_color = (0, 0, 255) if current_behavior in ("Agitado", "Dormindo", "Distraido") else (0, 255, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                label_text = f"{name_student} - {current_behavior}"
                rendered_tracks_count += 1

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2
                pad_x, pad_y = 6, 4
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, scale, thickness)

                tx = int(x_min)
                ty = int(y_min)
                top = ty - text_h - 2 * pad_y
                if top < 0:
                    top = ty

                cv2.rectangle(frame, (tx, top), (tx + text_w + 2 * pad_x, top + text_h + 2 * pad_y), box_color, -1)
                cv2.putText(frame, label_text, (tx + pad_x, top + text_h + pad_y - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
                        cv2.putText(frame, text, (10, y0 + int(i * 22 * debug_font)),
                                    cv2.FONT_HERSHEY_SIMPLEX, debug_font, (255, 255, 0), 2)

    last_rendered_frame_id = st.session_state.get("monitor_last_rendered_frame_id", -1)
    if frame_id != last_rendered_frame_id:
        st.session_state["monitor_last_rendered_frame_id"] = frame_id

    if known_face_encodings_norm is None or len(known_face_names) == 0:
        status_placeholder.warning(
            "Nenhum embedding de aluno foi carregado. "
            "Rode o processo de geracao de embeddings para habilitar o reconhecimento facial."
        )
    else:
        status_placeholder.empty()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.session_state["monitor_live_stats"] = {
        "detected_faces_count": detected_faces_count,
        "recognized_faces_count": recognized_faces_count,
        "rendered_tracks_count": rendered_tracks_count,
    }
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
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
    sims = cosine_similarity([emb], known_face_encodings_norm)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    second_best_score = -1.0
    if len(sims) > 1:
        second_best_score = float(np.partition(sims, -2)[-2])

    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    if face_area < 7000:
        acceptance_threshold = FACE_RECOGNITION_SMALL_THRESHOLD
    elif face_area < 14000:
        acceptance_threshold = FACE_RECOGNITION_MEDIUM_THRESHOLD
    else:
        acceptance_threshold = FACE_RECOGNITION_BASE_THRESHOLD

    margin = best_score - second_best_score if second_best_score >= 0 else best_score
    if best_score >= acceptance_threshold and margin >= FACE_RECOGNITION_MIN_MARGIN:
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

# ---------------- Detector em thread separada (IA fora do loop de render) ----------------
class DetectorWorker:
    """
    Roda YOLO (pose) + InsightFace em background, sempre no frame mais recente.
    Evita fila e mantém o vídeo "ao vivo".
    """
    def __init__(self, model_pose, model_face, device, min_inference_interval=0.18, pose_imgsz=960, pose_conf=0.22):
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
            self._latest_frame = frame
            if frame_id is None:
                self._latest_frame_id += 1
            else:
                self._latest_frame_id = frame_id

    def get_outputs(self):
        with self._lock:
            faces = self._last_faces
            results = self._last_results
        return results, faces

    def _run(self):
        while self._running:
            frame = None
            frame_id = -1
            with self._lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
                self._latest_frame = None
            if frame is None:
                time.sleep(0.003)
                continue
            if frame_id < 0:
                time.sleep(0.003)
                continue
            now = time.time()
            if now - self._last_inference_at < self._min_inference_interval:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.model_face.get(rgb)
            results = self.model_pose.predict(
                frame,
                show=False,
                device=self.device,
                verbose=False,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
                half=(self.device == "cuda"),
            )
            with self._lock:
                self._last_faces = faces
                self._last_results = results
            self._last_inference_at = time.time()

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

def classify_behavior(nose, ls, rs, le, re, lw, rw, threshold):
    """
    0:nose | 5-6: ombros (ls, rs) | 7-8: cotovelos (le, re) | 9-10: punhos (lw, rw)
    """
    cx = (ls[0] + rs[0]) / 2.0
    cy = (ls[1] + rs[1]) / 2.0
    s  = max(1.0, float(abs(ls[0] - rs[0])))   # escala ombro-a-ombro

    # --- MÃOS ALTAS -> Perguntando/Agitado (robusto à distância) ---
    shoulder_line = cy - 0.10 * s
    up_L = (
        lw[2] > threshold and le[2] > threshold and
        lw[1] < nose[1] and lw[1] < shoulder_line and lw[1] < le[1]
    )
    up_R = (
        rw[2] > threshold and re[2] > threshold and
        rw[1] < nose[1] and rw[1] < shoulder_line and rw[1] < re[1]
    )
    if up_L and up_R:
        return "Agitado" if abs(lw[0] - rw[0]) > 0.90 * s else "Perguntando"
    if up_L or up_R:
        return "Perguntando"

    # --- DORMINDO: cabeça baixa com apoio de braço/mão ---
    MIN_S_FOR_SLEEP = 28.0
    DY_COEF   = 0.14
    DX_COEF   = 0.35 if s >= 50 else 0.55   # tolera cabeça de lado se estiver longe
    ELB_NEAR  = 0.26
    WRIST_ELBOW_X_NEAR = 0.38
    WRIST_ELBOW_Y_NEAR = 0.30

    dy = nose[1] - cy
    dx = abs(nose[0] - cx)
    best_elbow = min(abs(nose[1] - le[1]), abs(nose[1] - re[1]))
    hands_low  = (lw[1] > cy - 0.08 * s) and (rw[1] > cy - 0.08 * s)
    elbows_low = (le[1] > cy - 0.12 * s) and (re[1] > cy - 0.12 * s)
    left_support = abs(lw[0] - le[0]) < WRIST_ELBOW_X_NEAR * s and abs(lw[1] - le[1]) < WRIST_ELBOW_Y_NEAR * s
    right_support = abs(rw[0] - re[0]) < WRIST_ELBOW_X_NEAR * s and abs(rw[1] - re[1]) < WRIST_ELBOW_Y_NEAR * s
    wrist_support = left_support or right_support

    if s >= MIN_S_FOR_SLEEP and hands_low and elbows_low:
        head_low_ok   = (dy > DY_COEF * s) and (dx < DX_COEF * s)
        elbow_near_ok = (best_elbow < ELB_NEAR * s) and (nose[1] > cy - 0.10 * s)
        if wrist_support and (head_low_ok or elbow_near_ok):
            return "Dormindo"
        if head_low_ok:
            return "Distraido"

    # --- fallback ---
    return "Atento"


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
        "Cadastro de Alunos": "🧑‍🎓    Cadastro de Alunos\nGerencie alunos e cadastros",
        "Monitoramento": "📹    Monitoramento\nAcompanhe as sessões em tempo real",
        "Gráficos": "📈    Gráficos\nVisualize dados e estatísticas",
        "Relatórios": "🗂️    Relatórios\nAcesse análises e relatórios observacionais",
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
    profile_col, logout_col = st.sidebar.columns([4.2, 1])
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
        logout_clicked = st.button("⇥", key="sidebar_logout_button", use_container_width=True)
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
        for key in ("authenticated", "cpf", "name", "city", "state", "role"):
            if key in st.query_params:
                del st.query_params[key]
        st.session_state.clear()
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

        # Tela de conclusão
        if registration_done:
            ultimo_nome = st.session_state.get("last_cad_nome", "")
            ultima_mat  = st.session_state.get("last_cad_matricula", "")
            ultimo_hash = st.session_state.get("last_cad_hash", "")
            embedding_status = st.session_state.get("last_embedding_status")
            embedding_message = st.session_state.get("last_embedding_message", "")

            if ultimo_hash and embedding_status is None:
                with st.spinner("Gerando embedding facial do aluno..."):
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
                if 'cadastro_cap' in st.session_state:
                    try: st.session_state.cadastro_cap.release()
                    except: pass
                    del st.session_state['cadastro_cap']

                for k in ["pose_index","img_index","cap_running","next_time","pose_done",
                          "registration_done","last_cad_nome","last_cad_matricula","last_cad_hash",
                          "last_embedding_status","last_embedding_message"]:
                    st.session_state.pop(k, None)

                st.session_state.cad_nome = ""
                st.session_state.cad_matricula = ""

                st.toast("Cadastro finalizado.")
                st.rerun()
            st.stop()

        pose_index = max(0, min(pose_index, len(POSES) - 1))
        pose_atual = POSES[pose_index]
        left_col, right_col = st.columns([1.05, 1.25], gap="large")
        with left_col:
            with st.container(border=True):
                _render_student_registration_card_header(
                    "👤",
                    "Dados do Aluno",
                    "Preencha os dados principais para iniciar o cadastro.",
                    "linear-gradient(180deg, #4BA3FF 0%, #1F73D8 100%)",
                )
                _render_student_registration_stepper(1)
                st.markdown(
                    "<div class='student-reg-info'>Selecione a turma e informe os dados do aluno para iniciar o cadastro.</div>",
                    unsafe_allow_html=True,
                )
                class_options = {
                    int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
                    for _, row in allowed_classes_df.iterrows()
                }
                selected_class_label = st.selectbox("Turma", list(class_options.values()))
                selected_class_id = next(key for key, value in class_options.items() if value == selected_class_label)

                form_col1, form_col2 = st.columns(2, gap="large")
                with form_col1:
                    nome_aluno = st.text_input("Nome do Aluno", key="cad_nome", placeholder="Digite o nome completo")
                with form_col2:
                    matricula = st.text_input("Matrícula", key="cad_matricula", placeholder="Digite a matrícula")

                nome_norm = _normalize_student_name(nome_aluno)
                matr_norm = _normalize_student_registration(matricula)
                registration_errors = _validate_student_registration_fields(nome_aluno, matricula) if nome_aluno or matricula else []
                registration_ready = bool(nome_norm and matr_norm and not registration_errors)

                if registration_errors:
                    for error in registration_errors:
                        st.warning(error)

                with st.expander("Configurações da captura (opcional)", expanded=False):
                    settings_col1, settings_col2 = st.columns(2, gap="large")
                    with settings_col1:
                        IMAGENS_POR_POSE = st.number_input("Número de poses por etapa", 1, 30, IMAGENS_POR_POSE, 1)
                    with settings_col2:
                        capture_interval = st.select_slider(
                            "Qualidade mínima",
                            options=[0.4, 0.8, 1.2, 1.6],
                            value=capture_interval,
                            format_func=lambda value: {
                                0.4: "Muito alta",
                                0.8: "Alta (recomendado)",
                                1.2: "Média",
                                1.6: "Econômica",
                            }[value],
                        )
                    prep_seconds = st.slider("Contagem inicial (segundos)", 0, 5, prep_seconds, 1)

                if not nome_norm and not matr_norm:
                    st.markdown(
                        "<div style='border-radius:14px; padding:0.9rem 1rem; margin-top:0.9rem; "
                        "background:linear-gradient(180deg, rgba(72,50,148,0.32) 0%, rgba(48,32,95,0.28) 100%); "
                        "border:1px solid rgba(121,94,255,0.14); color:#C9C3FF;'>"
                        "<strong>Passo 1 de 2</strong><br/>Após preencher os dados, clique em <strong>Iniciar captura</strong> para começar."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif not registration_ready:
                    st.info("Revise os campos informados para liberar a captura das imagens.")
                else:
                    st.success("Dados validados. A captura já pode ser iniciada.")

        with right_col:
            with st.container(border=True):
                _render_student_registration_card_header(
                    "📷",
                    "Controle da Captura",
                    "Acompanhe a pose atual e avance ao concluir cada etapa.",
                    "linear-gradient(180deg, #159957 0%, #0E7A46 100%)",
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
                        "Captura em andamento. Aguarde a conclusão automática desta pose.<br/>"
                        "Durante a coleta, faça pequenos movimentos para frente e para trás, varie levemente o ângulo "
                        "e, se possível, pegue pequenas diferenças de iluminação sem sair da pose atual."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif registration_ready:
                    st.markdown(
                        "<div class='student-reg-info'>Preenchimento validado. Você já pode iniciar a captura das poses.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='student-reg-info'>Preencha os dados do aluno para habilitar a captura das poses.</div>",
                        unsafe_allow_html=True,
                    )

                button_col1, button_col2, button_col3 = st.columns(3)
                with button_col1:
                    start_btn = st.button(
                        "▶️ Iniciar Captura",
                        disabled=(not registration_ready) or cap_running or pose_done,
                        use_container_width=True,
                    )
                with button_col2:
                    cancel_btn = st.button(
                        "⏹️ Cancelar",
                        disabled=(not registration_ready) or (not cap_running),
                        use_container_width=True,
                    )
                with button_col3:
                    next_btn = st.button(
                        "➡️ Próximo",
                        disabled=(not registration_ready)
                        or st.session_state.get("cap_running", False)
                        or not st.session_state.get("pose_done", False),
                        use_container_width=True,
                    )

                st.markdown(
                    "<div class='student-reg-lock'>Os botões serão habilitados automaticamente após a validação dos dados.</div>",
                    unsafe_allow_html=True,
                )

        if registration_ready:
            nome_criptografado = salvar_mapeamento(nome_norm, matr_norm)
            upsert_student(nome_criptografado, nome_norm, matr_norm, selected_class_id)

            os.makedirs(DATABASE_PATH, exist_ok=True)
            pasta_base = os.path.join(DATABASE_PATH, nome_criptografado)
            os.makedirs(pasta_base, exist_ok=True)
            for _pose in POSES:
                os.makedirs(os.path.join(pasta_base, _pose), exist_ok=True)

            # CSV fora da pasta 'alunos'
            if not os.path.exists(MAPPING_CSV):
                pd.DataFrame(columns=["nome", "matricula", "hash"]).to_csv(MAPPING_CSV, index=False)

            df = pd.read_csv(MAPPING_CSV)
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

            with right_col:
                if st.button("🔄 Atualizar embedding deste aluno", use_container_width=True):
                    with st.spinner("Processando embedding facial do aluno..."):
                        success, message = generate_student_embedding(
                            nome_criptografado,
                            nome_norm,
                            matr_norm,
                        )
                    if success:
                        st.success(message)
                        teardown_monitor_runtime()
                        st.session_state.pop("student_lookup", None)
                    else:
                        st.error(message)

            # Preview
            st.markdown("<div class='student-reg-camera-wrap'>", unsafe_allow_html=True)
            with st.container(border=True):
                _render_student_registration_card_header(
                    "📷",
                    "Câmera de captura",
                    "A visualização da câmera é exibida abaixo durante a coleta das poses.",
                    "linear-gradient(180deg, #5F49D6 0%, #4633A8 100%)",
                )
            stframe = st.empty()
            if 'cadastro_cap' not in st.session_state:
                st.session_state.cadastro_cap = cv2.VideoCapture(0)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap = st.session_state.cadastro_cap

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
                    st.session_state.last_cad_hash = nome_criptografado
                    st.session_state.last_embedding_status = None
                    st.session_state.last_embedding_message = ""
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
                    stframe.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", width=900)

                    if now >= (st.session_state.next_time or now):
                        timestamp    = get_local_now().strftime("%Y%m%d_%H%M%S%f")
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
                                st.session_state.last_cad_hash = nome_criptografado
                                st.session_state.last_embedding_status = None
                                st.session_state.last_embedding_message = ""
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
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=900)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.container(border=True):
                _render_student_registration_card_header(
                    "📷",
                    "Câmera de captura",
                    "A câmera será exibida aqui após a validação do cadastro.",
                    "linear-gradient(180deg, #5F49D6 0%, #4633A8 100%)",
                )
                preview_placeholder = st.empty()
                preview_placeholder.markdown(
                    """
                    <div class="student-reg-camera-placeholder">
                        <div class="student-reg-camera-icon">📷</div>
                        <div style="font-size:1.55rem; color:#E5E8F1; margin-bottom:0.45rem;">A visualização da câmera será exibida aqui</div>
                        <div style="font-size:1rem; color:#A6AFBD;">A captura é liberada automaticamente após validar o nome e a matrícula.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ------------------ MONITORAMENTO ------------------
    elif menu_option == "Monitoramento":
        if 'cadastro_cap' in st.session_state:
            try: st.session_state.cadastro_cap.release()
            except: pass
            del st.session_state['cadastro_cap']

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
        video_snapshot = _get_monitor_video_snapshot(current_session_preview)

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
        st.markdown("<div class='monitor-layout'>", unsafe_allow_html=True)
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
                    """, unsafe_allow_html=True)
                subject_options = {int(row["id"]): row["nome"] for _, row in subjects_df.iterrows()}
                default_subject_index = 0
                if selected_subject_id in subject_options:
                    default_subject_index = list(subject_options.keys()).index(selected_subject_id)
                disable_scope_inputs = bool(current_session_preview and current_session_preview["status"] == SESSION_STATUS_OPEN)
                selected_subject_label = st.selectbox(
                    "Disciplina da aula",
                    list(subject_options.values()),
                    index=default_subject_index,
                    disabled=disable_scope_inputs,
                )
                selected_subject_id = next(key for key, value in subject_options.items() if value == selected_subject_label)
                monitor_state["selected_subject_id"] = selected_subject_id

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
                selected_class_label = st.selectbox(
                    "Turma acompanhada",
                    list(class_options.values()),
                    index=default_class_index,
                    disabled=disable_scope_inputs,
                )
                selected_class_id = next(key for key, value in class_options.items() if value == selected_class_label)
                monitor_state["selected_class_id"] = selected_class_id

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

            if ui_state == "Em andamento":
                render_context_fragment(ui_state, selected_subject_label, selected_class_label, session_panel)
            else:
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
                    stop_system = st.button("■ Finalizar Sessão", disabled=True, use_container_width=True)
                elif ui_state == "Em andamento":
                    run_system = False
                    stop_system = st.button("■ Encerrar Monitoramento", use_container_width=True)
                else:
                    run_system = st.button("▶ Iniciar nova sessão", type="primary", use_container_width=True)
                    stop_system = st.button("■ Finalizar Sessão", disabled=True, use_container_width=True)

                st.markdown(
                    """
                    <div class="monitor-tip-box" style="margin-top:1rem;">
                        <div class="monitor-tip-title">Como funciona?</div>
                        Ao iniciar, o sistema começará a capturar e analisar os comportamentos automaticamente.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        current_session_id = monitor_state.get("session_id")
        current_session = get_monitoring_session_summary(current_session_id) if current_session_id else None

        if run_system:
            session_id = create_monitoring_session(
                user_context["teacher_id"],
                selected_subject_id,
                selected_class_id,
            )
            session_summary = get_monitoring_session_summary(session_id)
            st.session_state["student_lookup"] = get_student_lookup_for_scope(user_context)
            st.session_state["monitoring_state"] = {
                "session_id": session_id,
                "selected_subject_id": selected_subject_id,
                "selected_class_id": selected_class_id,
            }
            st.session_state.pop("last_closed_monitoring_session", None)
            st.session_state["episode_manager"] = BehaviorEpisodeManager(
                persist_callback=lambda **kwargs: insert_behavior_episode(
                    monitoring_session_id=session_id,
                    student_id=kwargs.get("id_student"),
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
            st.rerun()

        with monitor_right_col:
            with st.container(border=True):
                render_monitor_video_summary_fragment(current_session_id)
                if current_session:
                    with st.container(border=True):
                        monitor_status_placeholder = st.empty()
                        monitor_frame_placeholder = st.empty()
                        process_monitor_fragment(
                            school=school,
                            discipline=current_session["subject_name"],
                            user_name=user_name,
                            confidence_threshold=CONFIDENCE_THRESHOLD,
                            show_debug=show_debug,
                            debug_font=debug_font,
                            box_margin_ratio=BOX_MARGIN_RATIO,
                            show_unknown_boxes=show_unknown_boxes,
                            status_placeholder=monitor_status_placeholder,
                            frame_placeholder=monitor_frame_placeholder,
                        )
                else:
                    st.markdown(
                        """
                        <div class="monitor-placeholder">
                            <div>
                                <div class="monitor-placeholder-icon">📷</div>
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
        st.markdown("</div>", unsafe_allow_html=True)

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
            st.session_state["last_closed_monitoring_session"] = closed_session
            st.session_state.pop("monitor_waiting_since", None)
            st.session_state.pop("monitor_last_display_frame", None)
            st.session_state.pop("monitor_last_detector_frame_id", None)
            st.session_state.pop("monitor_last_rendered_frame_id", None)
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
