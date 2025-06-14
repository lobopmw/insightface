import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import datetime
from control_database import insert_count_behavior, df_behavior_charts, show_behavior_charts
from register_face_multi_images_avg import load_insightface_data
from sklearn.metrics.pairwise import cosine_similarity


from PIL import Image
from insightface.app import FaceAnalysis

def recognition_behavior():
    school = "Escola Estadual Criança Esperança"
    discipline = "Matemática"

    st.sidebar.image("../images/classroom1.jpg", use_container_width=True)
    user_name = st.session_state.get("name", "Usuário")
    st.sidebar.markdown(f"**{user_name}**")

    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.rerun()

    menu_option = st.sidebar.radio("Menu", ["Cadastro de Alunos", "Monitoramento", "Gráficos", "Tabela"])

    if menu_option == "Cadastro de Alunos":
        col_img1, col_img2, _ = st.columns([1,2,1])
        with col_img1:
            st.image("../images/faces.png", width=200)
        with col_img2:
            st.title("INFORMAÇÕES DO ALUNO")

    elif menu_option == "Monitoramento":
        col_img1, col_img2, _ = st.columns([1,4,1])
        with col_img1:
            st.image("../images/cam_IA.png", width=200)
        with col_img2:
            st.title("MONITORAMENTO")

        CONFIDENCE_THRESHOLD = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.5, 0.7)

        use_gpu = st.sidebar.checkbox("Usar GPU (CUDA)", value=True)
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        st.sidebar.write(f"Dispositivo: {device}")

        col1, col2 = st.sidebar.columns(2)
        run_system = col1.button("Iniciar Monitoramento")
        stop_system = col2.button("Parar Monitoramento")

        model = YOLO('yolo11n-pose.pt')
        behavior_tracker = {}
        BOX_MARGIN_RATIO = 0.2
        connections = [(0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10)]
        messege = st.empty()

        model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        model_face.prepare(ctx_id=0)

        known_face_encodings, known_face_names = load_insightface_data()
        st.write(f"[DEBUG] Quantidade de embeddings carregados: {len(known_face_encodings)}")
        st.write(f"[DEBUG] Nomes carregados: {known_face_names}")

        name_student = ""

        if not run_system and not stop_system:
            messege.info("Obs: O sistema irá monitorar os comportamentos dos alunos durante a aula. Inicie o monitoramento!")

        if run_system:
            messege.empty()
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            fps_limit = 7
            prev_time = 0

            while cap.isOpened():
                if time.time() - prev_time > 1.0 / fps_limit:
                    ret, frame = cap.read()
                    prev_time = time.time()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = model_face.get(rgb_frame)
                tracked_faces = {}
                name_student = "Desconhecido"

                for face in faces:

                    box = face.bbox.astype(int)

                    embedding = face.embedding

                   # Normaliza os embeddings
                    known_face_encodings_norm = known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

                    # Calcula similaridade de cosseno
                    similarities = cosine_similarity([embedding], known_face_encodings_norm)[0]
                    best_index = np.argmax(similarities)
                    best_score = similarities[best_index]

                    st.write(f"[DEBUG] Similaridade com {known_face_names[best_index]}: {best_score:.3f}")

                    if best_score > 0.45:  # Threshold recomendado para média de 5-10 imagens
                        name_student = known_face_names[best_index]
                    else:
                        name_student = "Desconhecido"

                    tracked_faces[name_student] = {
                        "location": (box[1], box[2], box[3], box[0]),
                        "last_seen": time.time()
                    }

                    # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    # cv2.putText(frame, name_student, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                results = model.predict(frame, show=False, device=device)

                if not results:
                    st.warning("Nenhuma pessoa detectada.")
                    continue

                for result in results:
                    if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
                        continue
                    keypoints = result.keypoints.data.cpu().numpy()

                    for id, person_keypoints in enumerate(keypoints):
                        if len(person_keypoints) == 0:
                            continue

                        person_id = name_student if name_student else f"Pessoa_{id}"
                        current_behavior = "Atento"

                        if person_keypoints.shape[0] > 10:
                            nose = person_keypoints[0]
                            ls, rs = person_keypoints[5], person_keypoints[6]
                            le, re = person_keypoints[7], person_keypoints[8]
                            lw, rw = person_keypoints[9], person_keypoints[10]

                            confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
                            if all(c > CONFIDENCE_THRESHOLD for c in confs):
                                def angle(a,b,c):
                                    ab, cb = a[:2]-b[:2], c[:2]-b[:2]
                                    return np.degrees(np.arccos(np.clip(np.dot(ab,cb)/(np.linalg.norm(ab)*np.linalg.norm(cb)), -1.0, 1.0)))

                                nose_y = nose[1]
                                shoulder_y = (ls[1] + rs[1]) / 2
                                angle_l = angle(ls, le, lw)
                                angle_r = angle(rs, re, rw)

                                if lw[1] < nose_y or rw[1] < nose_y:
                                    current_behavior = "Perguntando"
                                elif nose_y > shoulder_y and angle_l > 150 and angle_r > 150:
                                    current_behavior = "Dormindo"
                                elif nose_y > shoulder_y:
                                    current_behavior = "Escrevendo"

                        date = datetime.datetime.now().strftime("%Y-%m-%d")
                        current_time = datetime.datetime.now().strftime("%H:%M:%S")

                        if name_student not in behavior_tracker:
                            behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                        if behavior_tracker[name_student]["behavior"] != current_behavior and name_student != "Desconhecido":
                            insert_count_behavior(school, discipline, user_name, '12345', name_student,
                                                  behavior_tracker[name_student]["behavior"], date,
                                                  behavior_tracker[name_student]["start_time"], current_time)

                            behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                
                        for i, (x, y, c) in enumerate(person_keypoints):
                            if c > CONFIDENCE_THRESHOLD:
                                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                                cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        x_coords = [p[0] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                        y_coords = [p[1] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                        if x_coords and y_coords:
                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))
                            y_min = max(0, int(y_min - BOX_MARGIN_RATIO * (y_max - y_min)))

                            name_student = "Desconhecido"
                            for name, data in tracked_faces.items():
                                top, right, bottom, left = data["location"]
                                if x_min < right and x_max > left and y_min < bottom and y_max > top:
                                    name_student = name
                                    break

                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                            label = f"{name_student} - {current_behavior}"
                            cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                frame_resized = cv2.resize(frame, (1280, 720))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                stframe.image(Image.fromarray(frame_rgb), use_container_width=True)

            cap.release()
            cv2.destroyAllWindows()

        if stop_system:
            st.info("Monitoramento parado.")

    elif menu_option == "Gráficos":
        st.title("GRÁFICOS")
        show_behavior_charts()

    elif menu_option == "Tabela":
        col_img1, col_img2, _ = st.columns([1, 6, 1])
        with col_img1:
            st.image("../images/table.png", width=200)
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
