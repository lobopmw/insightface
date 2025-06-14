import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

def register_faces():
    # Inicializa modelo
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Caminhos absolutos
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'alunos'))
    EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy'))
    NAMES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'names.pkl'))

    os.makedirs(BASE_DIR, exist_ok=True)

    embeddings, names = [], []

    for student_name in os.listdir(BASE_DIR):
        student_path = os.path.join(BASE_DIR, student_name)
        if not os.path.isdir(student_path):
            continue

        student_embeddings = []

        for pose_folder in os.listdir(student_path):
            pose_path = os.path.join(student_path, pose_folder)
            if not os.path.isdir(pose_path):
                continue

            for img_file in os.listdir(pose_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(pose_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = app.get(img_rgb)

                    if faces:
                        emb = faces[0].embedding
                        student_embeddings.append(emb)
                        print(f"[OK] {student_name} - {pose_folder} - {img_file}")
                    else:
                        print(f"[X] Sem rosto: {student_name}/{pose_folder}/{img_file}")

        if student_embeddings:
            avg_emb = np.mean(student_embeddings, axis=0)
            embeddings.append(avg_emb)
            names.append(student_name)
            print(f"[✓] Média criada para {student_name} ({len(student_embeddings)} imagens)")
        else:
            print(f"[!] Nenhum rosto válido encontrado para {student_name}")

    np.save(EMB_PATH, np.array(embeddings))
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)

    print(f"[✔] Cadastro finalizado: {len(embeddings)} alunos salvos.")


def load_insightface_data():
    EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy'))
    NAMES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'names.pkl'))

    embeddings = np.load(EMB_PATH)
    with open(NAMES_PATH, "rb") as f:
        names = pickle.load(f)

    return embeddings, names
