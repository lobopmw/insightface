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


def load_mapping_students():
    if not os.path.exists(MAP_PATH):
        return pd.DataFrame(columns=["nome", "matricula", "hash"])

    df_map = pd.read_csv(MAP_PATH)
    if df_map.empty:
        return pd.DataFrame(columns=["nome", "matricula", "hash"])

    expected_columns = ["nome", "matricula", "hash"]
    for col in expected_columns:
        if col not in df_map.columns:
            df_map[col] = None

    return df_map[expected_columns].drop_duplicates(subset=["hash"], keep="last").reset_index(drop=True)


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


def _create_face_app():
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _iter_student_images(student_path):
    if not os.path.isdir(student_path):
        return

    for pose_folder in os.listdir(student_path):
        pose_path = os.path.join(student_path, pose_folder)
        if not os.path.isdir(pose_path):
            continue

        for img_file in os.listdir(pose_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                yield pose_folder, os.path.join(pose_path, img_file)


def _inspect_student_image_inventory(student_hash):
    student_path = os.path.join(ALUNOS_DIR, student_hash)
    if not os.path.isdir(student_path):
        return {
            "student_path": student_path,
            "folder_exists": False,
            "image_count": 0,
            "diagnostic": "Sem pasta de imagens cadastradas.",
        }

    image_count = sum(1 for _ in _iter_student_images(student_path))
    if image_count == 0:
        diagnostic = "Pasta encontrada, mas sem imagens válidas cadastradas."
    else:
        diagnostic = f"{image_count} imagem(ns) disponível(is) para processamento."

    return {
        "student_path": student_path,
        "folder_exists": True,
        "image_count": image_count,
        "diagnostic": diagnostic,
    }


def _extract_student_average_embedding(app, student_hash, nome):
    student_path = os.path.join(ALUNOS_DIR, student_hash)
    if not os.path.isdir(student_path):
        return None, 0, 0, f"Pasta do aluno não encontrada: {student_path}"

    student_embeddings = []
    images_found = 0

    for pose_folder, img_path in _iter_student_images(student_path):
        images_found += 1
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
            print(f"[OK] {nome} - {pose_folder} - {os.path.basename(img_path)}")
        else:
            print(f"[X] Sem rosto: {nome}/{pose_folder}/{os.path.basename(img_path)}")

    if not student_embeddings:
        return None, images_found, 0, f"Nenhum rosto válido encontrado para {nome}"

    student_embeddings_arr = np.asarray(student_embeddings, dtype=np.float32)
    avg_emb = student_embeddings_arr.mean(axis=0)
    return avg_emb, images_found, len(student_embeddings), None


def _upsert_local_embedding(name, embedding):
    if os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH):
        embeddings = np.load(EMB_PATH)
        with open(NAMES_PATH, "rb") as f:
            names = pickle.load(f)
    else:
        embeddings = np.empty((0, 512), dtype=np.float32)
        names = []

    names = [str(item) for item in names]
    emb_array = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    if name in names and len(embeddings) == len(names):
        idx = names.index(name)
        embeddings[idx] = emb_array[0]
    else:
        names.append(name)
        embeddings = emb_array if embeddings.size == 0 else np.vstack([embeddings, emb_array])

    np.save(EMB_PATH, np.asarray(embeddings, dtype=np.float32))
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)


def _load_existing_embedding_hashes():
    try:
        with managed_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT student_hash FROM face_embeddings")
                rows = cursor.fetchall()
    except psycopg2.Error as exc:
        print(f"❌ Erro ao consultar hashes de embeddings: {exc}")
        return set()

    return {str(row[0]) for row in rows if row and row[0]}


def get_embedding_maintenance_snapshot():
    df_map = load_mapping_students()
    if df_map.empty:
        return {
            "total_students": 0,
            "with_embeddings": 0,
            "pending_embeddings": 0,
            "students": df_map,
        }

    existing_hashes = _load_existing_embedding_hashes()
    snapshot_df = df_map.copy()
    snapshot_df["embedding_status"] = snapshot_df["hash"].astype(str).apply(
        lambda value: "Atualizado" if value in existing_hashes else "Pendente"
    )
    diagnostics = snapshot_df["hash"].astype(str).apply(_inspect_student_image_inventory)
    snapshot_df["image_count"] = diagnostics.apply(lambda item: item["image_count"])
    snapshot_df["embedding_diagnostic"] = snapshot_df.apply(
        lambda row: (
            "Embedding disponível no banco."
            if row["embedding_status"] == "Atualizado"
            else diagnostics.loc[row.name]["diagnostic"]
        ),
        axis=1,
    )
    return {
        "total_students": int(len(snapshot_df)),
        "with_embeddings": int((snapshot_df["embedding_status"] == "Atualizado").sum()),
        "pending_embeddings": int((snapshot_df["embedding_status"] == "Pendente").sum()),
        "students": snapshot_df,
    }


def generate_student_embedding(student_hash, name, matricula=None, app=None):
    ensure_face_embeddings_table()

    local_app = app or _create_face_app()
    avg_emb, images_found, valid_images, error_message = _extract_student_average_embedding(local_app, student_hash, name)
    if avg_emb is None:
        return False, error_message or "Não foi possível gerar o embedding do aluno."

    _upsert_local_embedding(name, avg_emb)

    records = [
        (
            student_hash,
            name,
            None if matricula is None else str(matricula),
            avg_emb.astype(float).tolist(),
            datetime.utcnow(),
        )
    ]

    try:
        bulk_upsert_embeddings(records)
        return (
            True,
            f"Embedding gerado com sucesso para {name} usando {valid_images} de {images_found} imagens e sincronizado com o banco.",
        )
    except psycopg2.Error:
        return (
            True,
            f"Embedding gerado para {name} usando {valid_images} de {images_found} imagens. Houve falha ao sincronizar com o banco, mas os arquivos locais foram atualizados.",
        )


def register_faces(student_hashes=None):
    app = _create_face_app()

    ensure_face_embeddings_table()

    df_map = load_mapping_students()
    if df_map.empty:
        print(f"❌ Arquivo de mapeamento não encontrado ou vazio: {MAP_PATH}")
        return {"processed": 0, "success": 0, "failed": 0, "failed_students": []}

    if student_hashes:
        requested_hashes = {str(item) for item in student_hashes}
        df_map = df_map[df_map["hash"].astype(str).isin(requested_hashes)].reset_index(drop=True)

    if df_map.empty:
        print("⚠️ Nenhum aluno correspondente foi encontrado para processamento.")
        return {"processed": 0, "success": 0, "failed": 0, "failed_students": []}

    print(f"[DEBUG] Alunos únicos encontrados no CSV: {len(df_map)}")
    print(df_map.head())

    records = []
    failed_students = []

    for _, row in df_map.iterrows():

        print(f"[DEBUG] Processando aluno: {row['nome']} - hash: {row['hash']}")

        nome = row['nome']
        matricula = row['matricula']
        hash_dir = row['hash']
        avg_emb, images_found, valid_images, error_message = _extract_student_average_embedding(app, hash_dir, nome)
        if avg_emb is None:
            print(f"[!] {error_message}")
            failed_students.append({"nome": nome, "hash": hash_dir, "motivo": error_message})
            continue

        _upsert_local_embedding(nome, avg_emb)
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
        print(f"[✓] Média criada para {nome} ({valid_images}/{images_found} imagens válidas)")

    if records:
        try:
            bulk_upsert_embeddings(records)
            print(f"[✔] Embeddings sincronizados com o PostgreSQL ({len(records)} registros).")
        except psycopg2.Error:
            print("⚠️ Falha ao atualizar o banco; arquivos locais foram gerados normalmente.")
        print(f"[✔] Cadastro finalizado: {len(records)} alunos salvos.")
    else:
        print("❌ Nenhum aluno foi salvo. Verifique as imagens.")

    return {
        "processed": int(len(df_map)),
        "success": int(len(records)),
        "failed": int(len(failed_students)),
        "failed_students": failed_students,
    }


def reprocess_pending_embeddings():
    snapshot = get_embedding_maintenance_snapshot()
    pending_df = snapshot["students"]
    if pending_df.empty:
        return {"processed": 0, "success": 0, "failed": 0, "failed_students": []}

    pending_hashes = pending_df.loc[pending_df["embedding_status"] == "Pendente", "hash"].astype(str).tolist()
    if not pending_hashes:
        return {"processed": 0, "success": 0, "failed": 0, "failed_students": []}

    return register_faces(student_hashes=pending_hashes)


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
