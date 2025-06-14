import streamlit as st
import cv2
import os
import time

POSES = ["frontal", "lateral_direita", "lateral_esquerda", "cabeca_baixa"]
IMAGENS_POR_POSE = 10
INTERVALO = 1  # segundos entre capturas

st.set_page_config(page_title="Captura com Vídeo ao Vivo", layout="centered")

# Sessão
st.session_state.setdefault("etapa", 0)
st.session_state.setdefault("capturadas", 0)
st.session_state.setdefault("capturando", False)
st.session_state.setdefault("finalizado", False)
st.session_state.setdefault("nome", "")
st.session_state.setdefault("ultima_captura", 0)

# Nome
st.title("Sistema de Captura por Pose (Vídeo ao Vivo)")
st.session_state.nome = st.text_input("Digite o nome do aluno:", value=st.session_state.nome).strip().lower().replace(" ", "_")

if not st.session_state.nome:
    st.warning("Digite o nome do aluno para iniciar.")
    st.stop()

pose_atual = POSES[st.session_state.etapa]
st.subheader(f"Pose atual: {pose_atual.replace('_', ' ').title()}")

barra = st.progress(st.session_state.capturadas / IMAGENS_POR_POSE)

video_display = st.empty()
start_button = st.button("Iniciar Captura", disabled=st.session_state.capturando or st.session_state.finalizado)

# Webcam e captura
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not cap.isOpened():
    st.error("Erro ao acessar a câmera.")
    st.stop()

if start_button:
    st.session_state.capturando = True
    st.session_state.ultima_captura = time.time()

while cap.isOpened() and not st.session_state.finalizado:
    ret, frame = cap.read()
    if not ret:
        continue

    # Desenha o retângulo para feedback visual
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    # Exibe vídeo ao vivo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_display.image(frame_rgb, channels="RGB")

    if st.session_state.capturando and st.session_state.capturadas < IMAGENS_POR_POSE:
        tempo_atual = time.time()
        if tempo_atual - st.session_state.ultima_captura >= INTERVALO:
            pasta = f"data/alunos/{st.session_state.nome}/{pose_atual}"
            os.makedirs(pasta, exist_ok=True)
            img_path = os.path.join(pasta, f"{str(st.session_state.capturadas+1).zfill(2)}.jpg")
            cv2.imwrite(img_path, frame)
            st.session_state.capturadas += 1
            st.session_state.ultima_captura = tempo_atual
            barra.progress(st.session_state.capturadas / IMAGENS_POR_POSE,
                           text=f"Capturando {st.session_state.capturadas} de {IMAGENS_POR_POSE}")

    if st.session_state.capturadas == IMAGENS_POR_POSE:
        st.session_state.capturando = False
        barra.progress(1.0, text="Captura concluída!")
        break

cap.release()

# Avançar para próxima pose
if not st.session_state.capturando and st.session_state.capturadas == IMAGENS_POR_POSE:
    if st.session_state.etapa < len(POSES) - 1:
        if st.button("Próxima Pose"):
            st.session_state.etapa += 1
            st.session_state.capturadas = 0
            st.session_state.ultima_captura = 0
            st.rerun()
    else:
        st.success("✅ Captura finalizada para todas as poses!")
        st.session_state.finalizado = True
