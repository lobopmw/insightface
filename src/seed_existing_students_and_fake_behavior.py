#!/usr/bin/env python3
"""
Carrega alunos existentes em data/mapeamento_alunos.csv para o banco e gera
episodios comportamentais ficticios para cada aluno, também exportando CSV.
"""

import argparse
import csv
import os
import pickle
import random
from datetime import date as date_cls
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MAP_PATH = os.path.join(DATA_DIR, "mapeamento_alunos.csv")
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")

BEHAVIOR_DURATION_RANGES: Dict[str, Tuple[int, int]] = {
    "Atento": (5 * 60, 12 * 60),
    "Perguntando": (10, 40),
    "Dormindo": (1 * 60, 5 * 60),
    "Distraido": (45, 3 * 60),
    "Agitado": (20, 90),
}

BEHAVIOR_WEIGHTS: Dict[str, float] = {
    "Atento": 0.65,
    "Perguntando": 0.16,
    "Distraido": 0.10,
    "Dormindo": 0.05,
    "Agitado": 0.04,
}


def load_mapping_rows() -> List[dict]:
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Arquivo nao encontrado: {MAP_PATH}")

    rows: List[dict] = []
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("nome") or "").strip()
            matricula = (row.get("matricula") or "").strip()
            hash_id = (row.get("hash") or "").strip()
            if not name or not hash_id:
                continue
            rows.append({"nome": name, "matricula": matricula, "hash": hash_id})

    # remove duplicatas por hash mantendo ultima ocorrencia
    unique = {}
    for r in rows:
        unique[r["hash"]] = r
    return list(unique.values())


def load_embeddings_by_name() -> Dict[str, np.ndarray]:
    if not os.path.exists(NAMES_PATH) or not os.path.exists(EMB_PATH):
        return {}

    with open(NAMES_PATH, "rb") as f:
        names = pickle.load(f)
    embeddings = np.load(EMB_PATH)

    if len(names) != len(embeddings):
        return {}

    data: Dict[str, np.ndarray] = {}
    for idx, nm in enumerate(names):
        if nm not in data:
            data[str(nm)] = embeddings[idx]
    return data


def db_connect_from_env():
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "insightface_db"),
        user=os.getenv("DB_USER", "insightface_user"),
        password=os.getenv("DB_PASSWORD", "insightface_password"),
    )
    register_vector(conn)
    return conn


def ensure_tables(conn):
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS students (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255)
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                student_hash VARCHAR(64) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                matricula VARCHAR(255),
                embedding vector(512) NOT NULL,
                updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS behavior_episode (
                id SERIAL PRIMARY KEY,
                school VARCHAR(255),
                discipline VARCHAR(255),
                teacher VARCHAR(255),
                student VARCHAR(255) NOT NULL,
                id_student VARCHAR(255),
                behavior VARCHAR(100) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                duration_seconds DOUBLE PRECISION NOT NULL,
                date DATE NOT NULL,
                source VARCHAR(20) DEFAULT 'simulated'
            )
            '''
        )
    conn.commit()


def upsert_students(conn, mapping_rows: List[dict]) -> int:
    values = [(r["hash"], r["nome"]) for r in mapping_rows]
    if not values:
        return 0
    with conn.cursor() as cursor:
        execute_values(
            cursor,
            '''
            INSERT INTO students (id, name)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
            ''',
            values,
        )
    conn.commit()
    return len(values)


def upsert_face_embeddings(conn, mapping_rows: List[dict], emb_by_name: Dict[str, np.ndarray]) -> int:
    values = []
    for r in mapping_rows:
        emb = emb_by_name.get(r["nome"])
        if emb is None:
            continue
        values.append((r["hash"], r["nome"], r["matricula"] or None, emb.astype(float).tolist(), datetime.utcnow()))

    if not values:
        return 0

    with conn.cursor() as cursor:
        execute_values(
            cursor,
            '''
            INSERT INTO face_embeddings (student_hash, name, matricula, embedding, updated_at)
            VALUES %s
            ON CONFLICT (student_hash) DO UPDATE
            SET name = EXCLUDED.name,
                matricula = EXCLUDED.matricula,
                embedding = EXCLUDED.embedding,
                updated_at = EXCLUDED.updated_at
            ''',
            values,
        )
    conn.commit()
    return len(values)


def _pick_behavior(rng: random.Random, behavior_weights: Dict[str, float], previous: str = "") -> str:
    behaviors = list(behavior_weights.keys())
    weights = list(behavior_weights.values())
    for _ in range(6):
        b = rng.choices(behaviors, weights=weights, k=1)[0]
        if b != previous:
            return b
    return b


def get_student_behavior_weights(student_index: int, total_students: int) -> Dict[str, float]:
    """
    Maioria dos alunos com perfil mais positivo (Atento alto).
    Pequena parcela com mais Distraido/Dormindo.
    """
    if total_students <= 0:
        return dict(BEHAVIOR_WEIGHTS)

    # Ultimo 25% dos alunos recebe perfil com mais risco de distracao/sono.
    cutoff = max(1, int(total_students * 0.75))
    if student_index >= cutoff:
        return {
            "Atento": 0.42,
            "Perguntando": 0.14,
            "Distraido": 0.22,
            "Dormindo": 0.14,
            "Agitado": 0.08,
        }

    return dict(BEHAVIOR_WEIGHTS)


def build_fake_episodes_for_student(
    rng: random.Random,
    student_name: str,
    student_id: str,
    base_start: datetime,
    total_minutes: int,
    school: str,
    discipline: str,
    teacher: str,
    behavior_weights: Dict[str, float],
) -> List[dict]:
    episodes: List[dict] = []
    current = base_start
    end_limit = base_start + timedelta(minutes=total_minutes)
    previous = ""

    while current < end_limit:
        behavior = _pick_behavior(rng, behavior_weights=behavior_weights, previous=previous)
        min_dur, max_dur = BEHAVIOR_DURATION_RANGES[behavior]
        dur = rng.randint(min_dur, max_dur)
        end_time = min(current + timedelta(seconds=dur), end_limit)

        episodes.append(
            {
                "school": school,
                "discipline": discipline,
                "teacher": teacher,
                "student": student_name,
                "id_student": student_id,
                "behavior": behavior,
                "start_time": current,
                "end_time": end_time,
                "duration_seconds": float((end_time - current).total_seconds()),
                "date": current.date(),
                "source": "simulated",
            }
        )

        previous = behavior
        current = end_time
        if current < end_limit:
            current += timedelta(seconds=rng.randint(0, 8))

    return episodes


def generate_fake_episodes_for_all_students(
    mapping_rows: List[dict],
    target_date: date_cls,
    seed: int,
    total_minutes: int,
    school: str,
    discipline: str,
    teacher: str,
) -> List[dict]:
    rng = random.Random(seed)
    all_eps: List[dict] = []

    start_base = datetime.combine(target_date, time(hour=8, minute=0))

    for idx, row in enumerate(mapping_rows):
        student_start = start_base + timedelta(minutes=idx * 2)
        student_weights = get_student_behavior_weights(idx, len(mapping_rows))
        eps = build_fake_episodes_for_student(
            rng=rng,
            student_name=row["nome"],
            student_id=row["hash"],
            base_start=student_start,
            total_minutes=total_minutes,
            school=school,
            discipline=discipline,
            teacher=teacher,
            behavior_weights=student_weights,
        )
        all_eps.extend(eps)

    return all_eps


def export_episodes_csv(episodes: List[dict], target_date: date_cls) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"behavior_episodes_fake_{target_date.strftime('%Y%m%d')}.csv")
    cols = [
        "school",
        "discipline",
        "teacher",
        "id_student",
        "student",
        "behavior",
        "start_time",
        "end_time",
        "duration_seconds",
        "date",
        "source",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for ep in episodes:
            row = dict(ep)
            row["start_time"] = ep["start_time"].strftime("%Y-%m-%d %H:%M:%S")
            row["end_time"] = ep["end_time"].strftime("%Y-%m-%d %H:%M:%S")
            row["date"] = ep["date"].strftime("%Y-%m-%d")
            writer.writerow(row)

    return csv_path


def insert_episodes(conn, episodes: List[dict], replace_for_date: bool = True) -> int:
    if not episodes:
        return 0

    with conn.cursor() as cursor:
        if replace_for_date:
            date_value = episodes[0]["date"]
            cursor.execute("DELETE FROM behavior_episode WHERE date = %s AND source = 'simulated'", (date_value,))

        values = [
            (
                ep["school"],
                ep["discipline"],
                ep["teacher"],
                ep["id_student"],
                ep["student"],
                ep["behavior"],
                ep["start_time"],
                ep["end_time"],
                ep["duration_seconds"],
                ep["date"],
                ep["source"],
            )
            for ep in episodes
        ]

        execute_values(
            cursor,
            '''
            INSERT INTO behavior_episode (
                school, discipline, teacher, id_student, student, behavior,
                start_time, end_time, duration_seconds, date, source
            )
            VALUES %s
            ''',
            values,
        )

    conn.commit()
    return len(episodes)


def parse_args():
    parser = argparse.ArgumentParser(description="Seed de alunos e episódios fictícios")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Data base YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-minutes", type=int, default=50)
    parser.add_argument("--school", default="Escola Estadual Criança Esperança")
    parser.add_argument("--discipline", default="Matemática")
    parser.add_argument("--teacher", default="Professor Simulado")
    parser.add_argument("--csv-only", action="store_true", help="Somente gerar CSV, sem gravar no banco")
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    mapping_rows = load_mapping_rows()
    emb_by_name = load_embeddings_by_name()

    episodes = generate_fake_episodes_for_all_students(
        mapping_rows=mapping_rows,
        target_date=target_date,
        seed=args.seed,
        total_minutes=args.total_minutes,
        school=args.school,
        discipline=args.discipline,
        teacher=args.teacher,
    )

    csv_path = export_episodes_csv(episodes, target_date)
    print(f"CSV gerado: {csv_path} ({len(episodes)} episódios)")

    if args.csv_only:
        return

    conn = None
    try:
        conn = db_connect_from_env()
        ensure_tables(conn)
        students_count = upsert_students(conn, mapping_rows)
        emb_count = upsert_face_embeddings(conn, mapping_rows, emb_by_name)
        ep_count = insert_episodes(conn, episodes, replace_for_date=True)
        print(f"students upsert: {students_count}")
        print(f"face_embeddings upsert: {emb_count}")
        print(f"behavior_episode insert: {ep_count}")
    except Exception as exc:
        print(f"Falha ao gravar no banco: {repr(exc)}")
        print("Os dados em CSV foram gerados normalmente.")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
