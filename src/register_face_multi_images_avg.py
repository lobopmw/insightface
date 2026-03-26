import os
from contextlib import contextmanager
from datetime import datetime
import cv2
import numpy as np
import pickle
import pandas as pd
import hashlib
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "insightface_db")
DB_USER = os.getenv("DB_USER", "insightface_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "insightface_password")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ALUNOS_DIR = os.path.join(DATA_DIR, 'alunos')
MAP_PATH = os.path.join(DATA_DIR, 'mapeamento_alunos.csv')
EMB_PATH = os.path.join(DATA_DIR, 'embeddings.npy')
NAMES_PATH = os.path.join(DATA_DIR, 'names.pkl')

@contextmanager
def managed_connection():
    """Abrir conexão com PostgreSQL já registrando o tipo vector."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    original_autocommit = conn.autocommit
    try:
        try:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            conn.autocommit = original_autocommit

        register_vector(conn)
        yield conn
    finally:
        conn.close()


def ensure_face_embeddings_table():
    """Garante extensão pgvector e tabela de embeddings."""
    try:
        with managed_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        student_hash VARCHAR(64) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        matricula VARCHAR(255),
                        embedding vector(512) NOT NULL,
                        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            conn.commit()
    except psycopg2.Error as exc:
        print(f"❌ Falha ao preparar tabela 'face_embeddings': {exc}")
        raise


def bulk_upsert_embeddings(records):
    """Insere ou atualiza embeddings de alunos no PostgreSQL."""
    if not records:
        return

    try:
        with managed_connection() as conn:
            with conn.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO face_embeddings (student_hash, name, matricula, embedding, updated_at)
                    VALUES %s
                    ON CONFLICT (student_hash) DO UPDATE
                    SET
                        name = EXCLUDED.name,
                        matricula = EXCLUDED.matricula,
                        embedding = EXCLUDED.embedding,
                        updated_at = EXCLUDED.updated_at
                    """,
                    records,
                )
            conn.commit()
    except psycopg2.Error as exc:
        print(f"❌ Erro ao salvar embeddings no banco: {exc}")
        raise


def get_hash(nome, matricula):
    return hashlib.sha256(f"{nome}_{matricula}".encode()).hexdigest()

def register_faces():
    # Inicializa modelo de face
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    ensure_face_embeddings_table()

    # Verifica CSV de mapeamento
    if not os.path.exists(MAP_PATH):
        print(f"❌ Arquivo de mapeamento não encontrado: {MAP_PATH}")
        return

    df_map = pd.read_csv(MAP_PATH)
    df_map = df_map.drop_duplicates(subset=["hash"], keep="last")

    print(f"[DEBUG] Alunos únicos encontrados no CSV: {len(df_map)}")
    print(df_map.head())


    embeddings, names = [], []
    records = []

    for _, row in df_map.iterrows():

        print(f"[DEBUG] Processando aluno: {row['nome']} - hash: {row['hash']}")

        nome = row['nome']
        matricula = row['matricula']
        hash_dir = row['hash']
        student_path = os.path.join(ALUNOS_DIR, hash_dir)

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
            student_embeddings_arr = np.asarray(student_embeddings, dtype=np.float32)
            avg_emb = student_embeddings_arr.mean(axis=0)
            embeddings.append(avg_emb)
            names.append(nome)
            matricula_str = None if pd.isna(matricula) else str(matricula)
            records.append(
                (
                    hash_dir,
                    nome,
                    matricula_str,
                    avg_emb.astype(float).tolist(),
                    datetime.utcnow(),
                )
            )
            print(f"[✓] Média criada para {nome} ({len(student_embeddings)} imagens)")
        else:
            print(f"[!] Nenhum rosto válido encontrado para {nome}")

    if embeddings and names:
        np.save(EMB_PATH, np.array(embeddings, dtype=np.float32))
        with open(NAMES_PATH, "wb") as f:
            pickle.dump(names, f)
        try:
            bulk_upsert_embeddings(records)
            print(f"[✔] Embeddings sincronizados com o PostgreSQL ({len(records)} registros).")
        except psycopg2.Error:
            print("⚠️ Falha ao atualizar o banco; arquivos locais foram gerados normalmente.")
        print(f"[✔] Cadastro finalizado: {len(embeddings)} alunos salvos.")
    else:
        print("❌ Nenhum aluno foi salvo. Verifique as imagens.")


def _load_embeddings_from_disk():
    if not os.path.exists(EMB_PATH) or not os.path.exists(NAMES_PATH):
        print("❌ Arquivos locais de embeddings não encontrados.")
        return np.empty((0, 512), dtype=np.float32), []

    embeddings = np.load(EMB_PATH)
    with open(NAMES_PATH, "rb") as f:
        names = pickle.load(f)

    return embeddings, names


def _load_embeddings_from_db():
    try:
        with managed_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT student_hash, name, embedding
                    FROM face_embeddings
                    ORDER BY student_hash
                    """
                )
                rows = cursor.fetchall()
    except psycopg2.Error as exc:
        print(f"❌ Erro ao carregar embeddings do banco: {exc}")
        return None, None

    if not rows:
        return np.empty((0, 512), dtype=np.float32), []

    names = []
    embeddings = []

    for _, name, embedding in rows:
        names.append(name)
        embeddings.append(np.array(embedding, dtype=np.float32))

    return np.array(embeddings, dtype=np.float32), names


def load_insightface_data():
    embeddings_db, names_db = _load_embeddings_from_db()
    if embeddings_db is not None:
        return embeddings_db, names_db

    print("⚠️ Utilizando embeddings locais em disco (arquivo .npy/.pkl).")
    return _load_embeddings_from_disk()


if __name__ == "__main__":
    register_faces()
