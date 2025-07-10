# import os
# import cv2
# import numpy as np
# import pickle
# from insightface.app import FaceAnalysis

# def register_faces():
#     # Inicializa modelo
#     app = FaceAnalysis(providers=['CPUExecutionProvider'])
#     app.prepare(ctx_id=-1, det_size=(640, 640))

#     # Caminhos absolutos
#     BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'alunos'))
#     EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy'))
#     NAMES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'names.pkl'))

#     os.makedirs(BASE_DIR, exist_ok=True)

#     embeddings, names = [], []

#     for student_name in os.listdir(BASE_DIR):
#         student_path = os.path.join(BASE_DIR, student_name)
#         if not os.path.isdir(student_path):
#             continue

#         student_embeddings = []

#         for pose_folder in os.listdir(student_path):
#             pose_path = os.path.join(student_path, pose_folder)
#             if not os.path.isdir(pose_path):
#                 continue

#             for img_file in os.listdir(pose_path):
#                 if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     img_path = os.path.join(pose_path, img_file)
#                     img = cv2.imread(img_path)
#                     if img is None:
#                         continue

#                     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     faces = app.get(img_rgb)

#                     if faces:
#                         emb = faces[0].embedding
#                         student_embeddings.append(emb)
#                         print(f"[OK] {student_name} - {pose_folder} - {img_file}")
#                     else:
#                         print(f"[X] Sem rosto: {student_name}/{pose_folder}/{img_file}")

#         if student_embeddings:
#             avg_emb = np.mean(student_embeddings, axis=0)
#             embeddings.append(avg_emb)
#             names.append(student_name)
#             print(f"[✓] Média criada para {student_name} ({len(student_embeddings)} imagens)")
#         else:
#             print(f"[!] Nenhum rosto válido encontrado para {student_name}")

#     np.save(EMB_PATH, np.array(embeddings))
#     with open(NAMES_PATH, "wb") as f:
#         pickle.dump(names, f)

#     print(f"[✔] Cadastro finalizado: {len(embeddings)} alunos salvos.")


# def load_insightface_data():
#     EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy'))
#     NAMES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'names.pkl'))

#     embeddings = np.load(EMB_PATH)
#     with open(NAMES_PATH, "rb") as f:
#         names = pickle.load(f)

#     return embeddings, names



################################ ATUALIZAÇÃO 10 07 2025 ##################################
import os
import cv2
import numpy as np
import pickle
import pandas as pd
import hashlib
from insightface.app import FaceAnalysis


def get_hash(nome, matricula):
    return hashlib.sha256(f"{nome}_{matricula}".encode()).hexdigest()

def register_faces():
    # Inicializa modelo de face
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Caminhos
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_dir = os.path.join(root_dir, 'data', 'alunos')
    map_path = os.path.join(root_dir, 'data', 'mapeamento_alunos.csv')
    emb_path = os.path.join(root_dir, 'data', 'embeddings.npy')
    names_path = os.path.join(root_dir, 'data', 'names.pkl')

    # Verifica CSV de mapeamento
    if not os.path.exists(map_path):
        print(f"❌ Arquivo de mapeamento não encontrado: {map_path}")
        return

    df_map = pd.read_csv(map_path)
    df_map = df_map.drop_duplicates(subset=["hash"], keep="last")

    print(f"[DEBUG] Alunos únicos encontrados no CSV: {len(df_map)}")
    print(df_map.head())


    embeddings, names = [], []

    for _, row in df_map.iterrows():

        print(f"[DEBUG] Processando aluno: {row['nome']} - hash: {row['hash']}")

        nome = row['nome']
        matricula = row['matricula']
        hash_dir = row['hash']
        student_path = os.path.join(base_dir, hash_dir)

        if not os.path.isdir(student_path):
            print(f"⚠️ Pasta não encontrada: {student_path}")
            continue

        student_embeddings = []

        for pose_folder in os.listdir(student_path):
            pose_path = os.path.join(student_path, pose_folder)
            if not os.path.isdir(pose_path):
                continue

            for img_file in os.listdir(pose_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(pose_path, img_file)
                    if not os.path.exists(img_path):
                        print(f"[X] Imagem não encontrada: {img_path}")
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[X] Falha ao ler imagem: {img_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = app.get(img_rgb)

                    if faces:
                        emb = faces[0].embedding
                        student_embeddings.append(emb)
                        print(f"[OK] {nome} - {pose_folder} - {img_file}")
                    else:
                        print(f"[X] Sem rosto: {nome}/{pose_folder}/{img_file}")

        if student_embeddings:
            avg_emb = np.mean(student_embeddings, axis=0)
            embeddings.append(avg_emb)
            names.append(nome)
            print(f"[✓] Média criada para {nome} ({len(student_embeddings)} imagens)")
        else:
            print(f"[!] Nenhum rosto válido encontrado para {nome}")

    if embeddings and names:
        np.save(emb_path, np.array(embeddings))
        with open(names_path, "wb") as f:
            pickle.dump(names, f)
        print(f"[✔] Cadastro finalizado: {len(embeddings)} alunos salvos.")
    else:
        print("❌ Nenhum aluno foi salvo. Verifique as imagens.")


def load_insightface_data():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    emb_path = os.path.join(root_dir, 'data', 'embeddings.npy')
    names_path = os.path.join(root_dir, 'data', 'names.pkl')

    if not os.path.exists(emb_path) or not os.path.exists(names_path):
        print("❌ Arquivos de embeddings ou nomes não encontrados.")
        return None, None

    embeddings = np.load(emb_path)
    with open(names_path, "rb") as f:
        names = pickle.load(f)

    return embeddings, names


if __name__ == "__main__":
    register_faces()
