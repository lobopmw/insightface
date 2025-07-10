# import cv2
# from ultralytics import YOLO
# import numpy as np
# import time
# import torch
# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import timedelta
# import datetime
# from control_database import insert_count_behavior, df_behavior_charts, show_behavior_charts
# from register_face_multi_images_avg import load_insightface_data
# from sklearn.metrics.pairwise import cosine_similarity
# import os


# from PIL import Image
# from insightface.app import FaceAnalysis

# image_path_classroom = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))
# image_path_faces = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/faces.png"))
# image_path_cam = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/cam_IA.png"))
# image_path_table = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/table.png"))

# def recognition_behavior():
#     school = "Escola Estadual Criança Esperança"
#     discipline = "Matemática"

    

#     st.sidebar.image(image_path_classroom, use_container_width=True)

#     user_name = st.session_state.get("name", "Usuário")
#     st.sidebar.markdown(f"**{user_name}**")

#     if st.sidebar.button("Sair"):
#         st.session_state.clear()
#         st.rerun()

#     menu_option = st.sidebar.radio("Menu", ["Cadastro de Alunos", "Monitoramento", "Gráficos", "Tabela"])

#     if menu_option == "Cadastro de Alunos":
#         col_img1, col_img2, _ = st.columns([1,2,1])
#         with col_img1:
#             st.image(image_path_faces, width=200)
#         with col_img2:
#             st.title("INFORMAÇÕES DO ALUNO")

#     elif menu_option == "Monitoramento":
#         col_img1, col_img2, _ = st.columns([1,4,1])
#         with col_img1:
#             st.image(image_path_cam, width=200)
#         with col_img2:
#             st.title("MONITORAMENTO")

#         CONFIDENCE_THRESHOLD = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.5, 0.7)

#         use_gpu = st.sidebar.checkbox("Usar GPU (CUDA)", value=True)
#         device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
#         st.sidebar.write(f"Dispositivo: {device}")

#         col1, col2 = st.sidebar.columns(2)
#         run_system = col1.button("Iniciar Monitoramento")
#         stop_system = col2.button("Parar Monitoramento")

#         model = YOLO('yolo11n-pose.pt')
#         behavior_tracker = {}
#         BOX_MARGIN_RATIO = 0.2
#         connections = [(0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10)]
#         messege = st.empty()

#         model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
#         model_face.prepare(ctx_id=0)

#         known_face_encodings, known_face_names = load_insightface_data()
#         st.write(f"[DEBUG] Quantidade de embeddings carregados: {len(known_face_encodings)}")
#         st.write(f"[DEBUG] Nomes carregados: {known_face_names}")

#         name_student = ""

#         if not run_system and not stop_system:
#             messege.info("Obs: O sistema irá monitorar os comportamentos dos alunos durante a aula. Inicie o monitoramento!")

#         if run_system:
#             messege.empty()
#             stframe = st.empty()
#             cap = cv2.VideoCapture(0)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#             fps_limit = 7
#             prev_time = 0

#             while cap.isOpened():
#                 if time.time() - prev_time > 1.0 / fps_limit:
#                     ret, frame = cap.read()
#                     prev_time = time.time()
#                 if not ret:
#                     break

#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 faces = model_face.get(rgb_frame)
#                 tracked_faces = {}
#                 name_student = "Desconhecido"

#                 for face in faces:

#                     box = face.bbox.astype(int)

#                     embedding = face.embedding

#                    # Normaliza os embeddings
#                     known_face_encodings_norm = known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
#                     embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

#                     # Calcula similaridade de cosseno
#                     similarities = cosine_similarity([embedding], known_face_encodings_norm)[0]
#                     best_index = np.argmax(similarities)
#                     best_score = similarities[best_index]

#                     st.write(f"[DEBUG] Similaridade com {known_face_names[best_index]}: {best_score:.3f}")

#                     if best_score > 0.45:  # Threshold recomendado para média de 5-10 imagens
#                         name_student = known_face_names[best_index]
#                     else:
#                         name_student = "Desconhecido"

#                     tracked_faces[name_student] = {
#                         "location": (box[1], box[2], box[3], box[0]),
#                         "last_seen": time.time()
#                     }

#                     # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                     # cv2.putText(frame, name_student, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#                 results = model.predict(frame, show=False, device=device)

#                 if not results:
#                     st.warning("Nenhuma pessoa detectada.")
#                     continue

#                 for result in results:
#                     if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
#                         continue
#                     keypoints = result.keypoints.data.cpu().numpy()

#                     for id, person_keypoints in enumerate(keypoints):
#                         if len(person_keypoints) == 0:
#                             continue

#                         person_id = name_student if name_student else f"Pessoa_{id}"
#                         current_behavior = "Atento"

#                         if person_keypoints.shape[0] > 10:
#                             nose = person_keypoints[0]
#                             ls, rs = person_keypoints[5], person_keypoints[6]
#                             le, re = person_keypoints[7], person_keypoints[8]
#                             lw, rw = person_keypoints[9], person_keypoints[10]

#                             confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
#                             if all(c > CONFIDENCE_THRESHOLD for c in confs):
#                                 def angle(a,b,c):
#                                     ab, cb = a[:2]-b[:2], c[:2]-b[:2]
#                                     return np.degrees(np.arccos(np.clip(np.dot(ab,cb)/(np.linalg.norm(ab)*np.linalg.norm(cb)), -1.0, 1.0)))

#                                 nose_y = nose[1]
#                                 shoulder_y = (ls[1] + rs[1]) / 2
#                                 angle_l = angle(ls, le, lw)
#                                 angle_r = angle(rs, re, rw)

#                                 # if lw[1] < nose_y or rw[1] < nose_y:
#                                 #     current_behavior = "Perguntando"
#                                 # elif nose_y > shoulder_y and angle_l > 150 and angle_r > 150:
#                                 #     current_behavior = "Dormindo"
#                                 # elif nose_y > shoulder_y:
#                                 #     current_behavior = "Escrevendo"

#                                 shoulder_center_x = (ls[0] + rs[0]) / 2
#                                 shoulder_y = (ls[1] + rs[1]) / 2
#                                 head_shift_x = abs(nose[0] - shoulder_center_x)
#                                 neck_angle = angle(ls, nose, rs)

#                                 # Dormindo: cabeça baixa e centralizada, pouco movimento nos braços
#                                 if nose[1] > shoulder_y and head_shift_x < 30 and neck_angle < 30:
#                                     current_behavior = "Dormindo"

#                                 # Perguntando: mãos acima da cabeça
#                                 elif lw[1] < nose[1] or rw[1] < nose[1]:
#                                     current_behavior = "Perguntando"

#                                 # Escrevendo: cabeça baixa mas com braços movimentados
#                                 elif nose[1] > shoulder_y:
#                                     current_behavior = "Escrevendo"

#                                 else:
#                                     current_behavior = "Atento"

#                         date = datetime.datetime.now().strftime("%Y-%m-%d")
#                         current_time = datetime.datetime.now().strftime("%H:%M:%S")

#                         if name_student not in behavior_tracker:
#                             behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

#                         if behavior_tracker[name_student]["behavior"] != current_behavior and name_student != "Desconhecido":
#                             insert_count_behavior(school, discipline, user_name, '12345', name_student,
#                                                   behavior_tracker[name_student]["behavior"], date,
#                                                   behavior_tracker[name_student]["start_time"], current_time)

#                             behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                
#                         for i, (x, y, c) in enumerate(person_keypoints):
#                             if c > CONFIDENCE_THRESHOLD:
#                                 cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
#                                 cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#                         x_coords = [p[0] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
#                         y_coords = [p[1] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
#                         if x_coords and y_coords:
#                             x_min, x_max = int(min(x_coords)), int(max(x_coords))
#                             y_min, y_max = int(min(y_coords)), int(max(y_coords))
#                             y_min = max(0, int(y_min - BOX_MARGIN_RATIO * (y_max - y_min)))

#                             name_student = "Desconhecido"
#                             for name, data in tracked_faces.items():
#                                 top, right, bottom, left = data["location"]
#                                 if x_min < right and x_max > left and y_min < bottom and y_max > top:
#                                     name_student = name
#                                     break

#                             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
#                             label = f"{name_student} - {current_behavior}"
#                             cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#                 frame_resized = cv2.resize(frame, (1280, 720))
#                 frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#                 stframe.image(Image.fromarray(frame_rgb), use_container_width=True)

#             cap.release()
#             cv2.destroyAllWindows()

#         if stop_system:
#             st.info("Monitoramento parado.")

#     elif menu_option == "Gráficos":
#         st.title("GRÁFICOS")
#         show_behavior_charts()

#     elif menu_option == "Tabela":
#         col_img1, col_img2, _ = st.columns([1, 6, 1])
#         with col_img1:
#             st.image(image_path_table, width=200)
#         with col_img2:
#             st.title("INFORMAÇÕES")

#         df = df_behavior_charts()
#         if df.empty:
#             st.warning("Nenhum dado registrado.")
#             return

#         today = datetime.datetime.now()
#         selected_date = st.date_input("Selecione a Data", value=today,
#                                        min_value=today - timedelta(days=365),
#                                        max_value=today + timedelta(days=365))

#         selected_disciplines = st.multiselect("Filtrar por Disciplinas", df['Disciplina'].unique().tolist())
#         selected_behaviors = st.multiselect("Filtrar por Comportamentos", df['Comportamento'].unique().tolist())

#         df['Data'] = pd.to_datetime(df['Data']).dt.date
#         filtered_df = df[df['Data'] == selected_date]

#         if selected_disciplines:
#             filtered_df = filtered_df[filtered_df['Disciplina'].isin(selected_disciplines)]
#         if selected_behaviors:
#             filtered_df = filtered_df[filtered_df['Comportamento'].isin(selected_behaviors)]

#         if filtered_df.empty:
#             st.warning("Nenhum dado encontrado para os filtros selecionados.")
#         else:
#             st.dataframe(filtered_df, use_container_width=True)


#################################### ATUALIZAÇÃO 09/07/2025 #################################################################
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
import os
from PIL import Image
from insightface.app import FaceAnalysis
import warnings
import hashlib
from utils_criptografia import salvar_mapeamento

warnings.filterwarnings("ignore", category=FutureWarning)

image_path_classroom = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))
image_path_faces = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/faces.png"))
image_path_cam = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/cam_IA.png"))
image_path_table = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/table.png"))

def classify_behavior(nose, ls, rs, le, re, lw, rw, threshold):
    
    shoulder_y = (ls[1] + rs[1]) / 2
    wrist_y = min(lw[1], rw[1])
    dist_nose_to_wrist = min(abs(nose[1] - lw[1]), abs(nose[1] - rw[1]))

    # Distância horizontal entre os pulsos (pode indicar agitação se muito grande)
    wrist_distance = abs(lw[0] - rw[0])

    # Se ambas as mãos estiverem acima do nariz
    if lw[1] < nose[1] and rw[1] < nose[1]:
        # Se estiverem muito afastadas → agitado
        if abs(lw[0] - rw[0]) > 200:
            return "Agitado"
        else:
            return "Perguntando"
        
    # Se um dos pulsos estiver acima do nariz
    if lw[1] < nose[1] or rw[1] < nose[1]:
        return "Perguntando"

    # Se nariz estiver distância adequada dos pulsos: escrevendo
    if 90 < dist_nose_to_wrist < 250:
        return "Escrevendo"

    # Se nariz estiver abaixo dos ombros e mão muito próxima do nariz: dormindo
    if nose[1] > shoulder_y and dist_nose_to_wrist <= 80:
        return "Dormindo"


    # Se nariz estiver acima da linha dos ombros: atento
    if nose[1] < shoulder_y - 15:
        return "Atento"

    return "Atento"

def criptografar_nome_matricula(nome, matricula):
    return hashlib.sha256(f"{nome}_{matricula}".encode()).hexdigest()

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

    if menu_option == "Cadastro de Alunos":
        st.title("📸 Cadastro de Alunos com Captura Guiada")

        disciplinas = ["Matemática", "Português", "História", "Geografia", "Ciências"]
        disciplina = st.selectbox("📘 Selecione a Disciplina:", disciplinas)
        nome_aluno = st.text_input("Nome do Aluno:")
        matricula = st.text_input("Matrícula do Aluno:")

        POSES = ["frontal", "lateral_direita", "lateral_esquerda", "cabeca_baixa"]
        IMAGENS_POR_POSE = 10
        DATABASE_PATH = "data/alunos"

        if nome_aluno and matricula:
            nome_criptografado = salvar_mapeamento(nome_aluno, matricula)
            pasta_base = os.path.join(DATABASE_PATH, nome_criptografado)
            os.makedirs(pasta_base, exist_ok=True)

            # Atualiza CSV de mapeamento
            mapeamento_path = os.path.join(DATABASE_PATH, "mapeamento_alunos.csv")
            if not os.path.exists(mapeamento_path):
                pd.DataFrame(columns=["nome", "matricula", "hash"]).to_csv(mapeamento_path, index=False)

            df = pd.read_csv(mapeamento_path)
            if not ((df["nome"] == nome_aluno) & (df["matricula"] == matricula)).any():
                novo = pd.DataFrame([{"nome": nome_aluno, "matricula": matricula, "hash": nome_criptografado}])
                df = pd.concat([df, novo], ignore_index=True)
                df.to_csv(mapeamento_path, index=False)

            pose_index = st.session_state.get("pose_index", 0)
            img_index = st.session_state.get("img_index", 0)
            pose_atual = POSES[pose_index]

            st.subheader(f"👉 Pose atual: **{pose_atual.replace('_', ' ').title()}** ({img_index + 1}/{IMAGENS_POR_POSE})")

            stframe = st.empty()
            botao_capturar = st.button("📸 Capturar Imagem")

            if 'cadastro_cap' not in st.session_state:
                st.session_state.cadastro_cap = cv2.VideoCapture(0)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.cadastro_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            cap = st.session_state.cadastro_cap
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", width=480)

                if botao_capturar:
                    pasta_pose = os.path.join(pasta_base, pose_atual)
                    os.makedirs(pasta_pose, exist_ok=True)

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    nome_arquivo = f"{pose_atual}_{timestamp}.jpg"
                    caminho = os.path.join(pasta_pose, nome_arquivo)
                    cv2.imwrite(caminho, frame)
                    st.success(f"Imagem salva: {nome_arquivo}")

                    img_index += 1
                    if img_index >= IMAGENS_POR_POSE:
                        img_index = 0
                        pose_index += 1

                    st.session_state["img_index"] = img_index
                    st.session_state["pose_index"] = pose_index

                    if pose_index >= len(POSES):
                        st.success("✅ Todas as imagens foram capturadas com sucesso!")
                        st.balloons()
                        st.session_state["pose_index"] = 0
                        st.session_state["img_index"] = 0

        else:
            st.warning("Preencha o nome e matrícula do aluno para iniciar a captura.")


    elif menu_option == "Monitoramento":
        col_img1, col_img2, _ = st.columns([1,4,1])
        with col_img1:
            st.image(image_path_cam, width=200)
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
        model_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        model_face.prepare(ctx_id=0)

        known_face_encodings, known_face_names = load_insightface_data()
        # st.write(f"[DEBUG] Quantidade de embeddings carregados: {len(known_face_encodings)}")
        # st.write(f"[DEBUG] Nomes carregados: {known_face_names}")

        messege = st.empty()

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
                face_boxes = []

                for face in faces:
                    box = face.bbox.astype(int)
                    face_boxes.append(box)
                    embedding = face.embedding
                    known_face_encodings_norm = known_face_encodings / (np.linalg.norm(known_face_encodings, axis=1, keepdims=True) + 1e-6)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                    similarities = cosine_similarity([embedding], known_face_encodings_norm)[0]
                    best_index = np.argmax(similarities)
                    best_score = similarities[best_index]

                    if best_score > 0.45:
                        name_student = known_face_names[best_index]
                    else:
                        name_student = "Desconhecido"

                    tracked_faces[name_student] = {
                        "location": (box[1], box[2], box[3], box[0]),
                        "last_seen": time.time()
                    }

                results = model.predict(frame, show=False, device=device, verbose=False)

                for result in results:
                    if not hasattr(result, 'keypoints') or len(result.keypoints) == 0:
                        continue
                    keypoints = result.keypoints.data.cpu().numpy()

                    for id, person_keypoints in enumerate(keypoints):
                        if len(person_keypoints) == 0:
                            continue

                        current_behavior = "Atento"

                        if person_keypoints.shape[0] > 10:
                            nose = person_keypoints[0]
                            ls, rs = person_keypoints[5], person_keypoints[6]
                            le, re = person_keypoints[7], person_keypoints[8]
                            lw, rw = person_keypoints[9], person_keypoints[10]

                            confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
                            if all(c > CONFIDENCE_THRESHOLD for c in confs):
                                current_behavior = classify_behavior(nose, ls, rs, le, re, lw, rw, CONFIDENCE_THRESHOLD)
                    #   ########################################################################################
                    #     # Coordenadas Y para depuração
                    #     cv2.putText(frame, f"Nose Y: {int(nose[1])}", (int(nose[0]), int(nose[1]) - 40),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    #     cv2.putText(frame, f"LS Y: {int(ls[1])}", (int(ls[0]), int(ls[1]) - 40),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    #     cv2.putText(frame, f"RS Y: {int(rs[1])}", (int(rs[0]), int(rs[1]) - 40),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    #     cv2.putText(frame, f"LW Y: {int(lw[1])}", (int(lw[0]), int(lw[1]) + 20),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    #     cv2.putText(frame, f"RW Y: {int(rw[1])}", (int(rw[0]), int(rw[1]) + 20),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    #     # Mostra também o valor médio dos ombros e a distância
                    #     shoulder_y = (ls[1] + rs[1]) / 2
                    #     dist_nose_wrist = min(abs(nose[1] - lw[1]), abs(nose[1] - rw[1]))
                    #     cv2.putText(frame, f"Shoulder Y: {int(shoulder_y)}", (10, 30),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                    #     cv2.putText(frame, f"Dist Nose-Wrist: {int(dist_nose_wrist)}", (10, 60),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

                    #     #####################################################################################

                        x_coords = [p[0] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                        y_coords = [p[1] for p in person_keypoints if p[2] > CONFIDENCE_THRESHOLD]
                        if not x_coords or not y_coords:
                            continue

                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        y_min = max(0, int(y_min - BOX_MARGIN_RATIO * (y_max - y_min)))

                        # Verifica interseção com rosto detectado
                        name_student = "Desconhecido"
                        for box in face_boxes:
                            fx1, fy1, fx2, fy2 = box
                            if x_min < fx2 and x_max > fx1 and y_min < fy2 and y_max > fy1:
                                for name, data in tracked_faces.items():
                                    top, right, bottom, left = data["location"]
                                    if fx1 == left and fx2 == right and fy1 == top and fy2 == bottom:
                                        name_student = name
                                        break

                        date = datetime.datetime.now().strftime("%Y-%m-%d")
                        current_time = datetime.datetime.now().strftime("%H:%M:%S")

                        if name_student not in behavior_tracker:
                            behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                        if behavior_tracker[name_student]["behavior"] != current_behavior and name_student != "Desconhecido":
                            insert_count_behavior(school, discipline, user_name, '12345', name_student,
                                                  behavior_tracker[name_student]["behavior"], date,
                                                  behavior_tracker[name_student]["start_time"], current_time)

                            behavior_tracker[name_student] = {"behavior": current_behavior, "start_time": current_time}

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"{name_student} -> {current_behavior}"
                        cv2.putText(frame, label, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
