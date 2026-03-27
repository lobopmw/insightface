import datetime
import io
import os
import tempfile
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "insightface_db")
DB_USER = os.getenv("DB_USER", "insightface_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "insightface_password")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
)

DEFAULT_SCHOOL_NAME = "Escola Estadual Criança Esperança"
SESSION_STATUS_OPEN = "em_andamento"
SESSION_STATUS_CLOSED = "encerrada"
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Araguaina")


def get_local_now() -> datetime.datetime:
    return datetime.datetime.now(ZoneInfo(APP_TIMEZONE)).replace(tzinfo=None)


def _ensure_schema(cursor) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            cpf VARCHAR(11) UNIQUE NOT NULL,
            nome VARCHAR(255) NOT NULL,
            password VARCHAR(255) NOT NULL,
            cidade VARCHAR(255),
            estado VARCHAR(2),
            email VARCHAR(255),
            role VARCHAR(20) NOT NULL DEFAULT 'professor',
            ativo BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_users_cpf UNIQUE(cpf)
        )
        """
    )
    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255)")
    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'professor'")
    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS ativo BOOLEAN NOT NULL DEFAULT TRUE")
    cursor.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )
    cursor.execute("UPDATE users SET role = COALESCE(NULLIF(role, ''), 'professor')")
    cursor.execute("UPDATE users SET ativo = COALESCE(ativo, TRUE)")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            nome VARCHAR(255) NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id SERIAL PRIMARY KEY,
            nome VARCHAR(255) NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            nome VARCHAR(255) NOT NULL,
            identificador VARCHAR(255),
            CONSTRAINT uq_classes_nome_identificador UNIQUE (nome, identificador)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS teacher_subject_class (
            id SERIAL PRIMARY KEY,
            teacher_id INTEGER NOT NULL REFERENCES teachers(id) ON DELETE CASCADE,
            subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
            CONSTRAINT uq_teacher_subject_class UNIQUE (teacher_id, subject_id, class_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            matricula VARCHAR(255),
            class_id INTEGER REFERENCES classes(id) ON DELETE SET NULL,
            ativo BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS matricula VARCHAR(255)")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS class_id INTEGER REFERENCES classes(id) ON DELETE SET NULL")
    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS ativo BOOLEAN NOT NULL DEFAULT TRUE")
    cursor.execute(
        "ALTER TABLE students ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )

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

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS monitoring_sessions (
            id SERIAL PRIMARY KEY,
            teacher_id INTEGER NOT NULL REFERENCES teachers(id) ON DELETE RESTRICT,
            subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE RESTRICT,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE RESTRICT,
            session_date DATE NOT NULL,
            start_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            end_time TIMESTAMP WITHOUT TIME ZONE,
            status VARCHAR(20) NOT NULL DEFAULT 'em_andamento',
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS behavior_log (
            id SERIAL PRIMARY KEY,
            school VARCHAR(255),
            discipline VARCHAR(255),
            teacher VARCHAR(255),
            student VARCHAR(255),
            id_student VARCHAR(255),
            behavior VARCHAR(100),
            count INTEGER DEFAULT 0,
            date VARCHAR(10),
            start_time VARCHAR(8),
            end_time VARCHAR(8),
            CONSTRAINT uq_behavior_log UNIQUE(student, behavior, date)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS behavior_episode (
            id SERIAL PRIMARY KEY,
            monitoring_session_id INTEGER REFERENCES monitoring_sessions(id) ON DELETE CASCADE,
            student_id VARCHAR(255) REFERENCES students(id) ON DELETE SET NULL,
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
            source VARCHAR(20) DEFAULT 'realtime',
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS monitoring_session_id INTEGER REFERENCES monitoring_sessions(id) ON DELETE CASCADE"
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS student_id VARCHAR(255) REFERENCES students(id) ON DELETE SET NULL"
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_session_student
        ON behavior_episode (monitoring_session_id, student_id, start_time)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_date
        ON behavior_episode (student, date)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_start_time
        ON behavior_episode (student, start_time)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_monitoring_sessions_scope
        ON monitoring_sessions (teacher_id, subject_id, class_id, session_date)
        """
    )


@contextmanager
def connect_database():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    try:
        register_vector(conn)
        with conn.cursor() as cursor:
            _ensure_schema(cursor)
            conn.commit()
        cursor = conn.cursor()
        yield conn, cursor
    finally:
        conn.close()


def user_table():
    try:
        metadata = MetaData()
        Table(
            "users",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("cpf", String(11), unique=True, nullable=False),
            Column("nome", String(255), nullable=False),
            Column("password", String(255), nullable=False),
            Column("cidade", String(255)),
            Column("estado", String(2)),
            Column("email", String(255)),
            Column("role", String(20), nullable=False, default="professor"),
            Column("ativo", Boolean, nullable=False, default=True),
            Column("created_at", DateTime, nullable=False),
            UniqueConstraint("cpf", name="uq_users_cpf"),
        )
        metadata.create_all(engine)
        with connect_database():
            pass
    except SQLAlchemyError as exc:
        print(f"Erro ao preparar tabela de usuarios: {exc}")


def _ensure_teacher_profile(conn, user_id: int, nome: str, role: str):
    if role != "professor":
        return None

    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM teachers WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            """
            INSERT INTO teachers (user_id, nome)
            VALUES (%s, %s)
            RETURNING id
            """,
            (user_id, nome),
        )
        teacher_id = cursor.fetchone()[0]
        conn.commit()
        return teacher_id


def get_user_by_cpf(cpf: str):
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            SELECT id, nome, cpf, password, cidade, estado, role, ativo
            FROM users
            WHERE cpf = %s
            """,
            (cpf,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        user = {
            "id": row[0],
            "nome": row[1],
            "cpf": row[2],
            "password": row[3],
            "cidade": row[4],
            "estado": row[5],
            "role": row[6] or "professor",
            "ativo": bool(row[7]),
        }
        user["teacher_id"] = _ensure_teacher_profile(conn, user["id"], user["nome"], user["role"])
        return user


def get_user_context(cpf: str):
    user = get_user_by_cpf(cpf)
    if not user:
        return None

    return {
        "user_id": user["id"],
        "teacher_id": user.get("teacher_id"),
        "cpf": user["cpf"],
        "name": user["nome"],
        "city": user["cidade"],
        "state": user["estado"],
        "role": (user["role"] or "professor").lower(),
        "active": user["ativo"],
    }


def registrar_usuario(cpf, nome, hashed_password, cidade, estado, role="professor", email=None):
    try:
        with connect_database() as (conn, cursor):
            cursor.execute("SELECT COUNT(*) FROM users WHERE cpf = %s", (cpf,))
            result = cursor.fetchone()[0]
            if result and result > 0:
                return "cpf_exists"

            cursor.execute(
                """
                INSERT INTO users (cpf, nome, password, cidade, estado, email, role, ativo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
                RETURNING id
                """,
                (cpf, nome, hashed_password, cidade, estado, email, role,),
            )
            user_id = cursor.fetchone()[0]
            _ensure_teacher_profile(conn, user_id, nome, role)
            conn.commit()
        return "ok"
    except IntegrityError:
        return "cpf_exists"
    except SQLAlchemyError as exc:
        print(f"Erro ao registrar usuario: {exc}")
        return "error"


def list_all_users():
    with connect_database() as (_, cursor):
        cursor.execute(
            """
            SELECT id, nome, cpf, cidade, estado, email, role, ativo, created_at
            FROM users
            ORDER BY nome ASC, id ASC
            """
        )
        rows = cursor.fetchall()

    return pd.DataFrame(
        rows,
        columns=["id", "nome", "cpf", "cidade", "estado", "email", "role", "ativo", "created_at"],
    )


def reset_user_password(user_id: int, hashed_password: str):
    try:
        with connect_database() as (conn, cursor):
            cursor.execute(
                """
                UPDATE users
                SET password = %s
                WHERE id = %s
                """,
                (hashed_password, user_id),
            )
            updated_rows = cursor.rowcount
            conn.commit()
        return "ok" if updated_rows else "not_found"
    except SQLAlchemyError as exc:
        print(f"Erro ao redefinir senha do usuario {user_id}: {exc}")
        return "error"


def upsert_student(student_id: str, name: str, matricula: str | None = None, class_id: int | None = None):
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO students (id, name, matricula, class_id, ativo)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (id)
            DO UPDATE SET
                name = EXCLUDED.name,
                matricula = COALESCE(EXCLUDED.matricula, students.matricula),
                class_id = COALESCE(EXCLUDED.class_id, students.class_id),
                ativo = TRUE
            """,
            (student_id, name, matricula, class_id),
        )
        conn.commit()


def list_teacher_assignments(teacher_id: int):
    query = """
        SELECT
            tsc.id,
            s.id AS subject_id,
            s.nome AS subject_name,
            c.id AS class_id,
            c.nome AS class_name,
            COALESCE(c.identificador, '') AS class_identifier
        FROM teacher_subject_class tsc
        JOIN subjects s ON s.id = tsc.subject_id
        JOIN classes c ON c.id = tsc.class_id
        WHERE tsc.teacher_id = %(teacher_id)s
        ORDER BY s.nome, c.nome, c.identificador NULLS FIRST
    """
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"teacher_id": teacher_id})


def list_subjects_for_user(user_context: dict):
    if user_context["role"] == "admin":
        query = "SELECT id, nome FROM subjects ORDER BY nome"
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    assignments = list_teacher_assignments(user_context["teacher_id"])
    if assignments.empty:
        return pd.DataFrame(columns=["id", "nome"])

    subjects = assignments[["subject_id", "subject_name"]].drop_duplicates().rename(
        columns={"subject_id": "id", "subject_name": "nome"}
    )
    return subjects.sort_values("nome").reset_index(drop=True)


def list_classes_for_user(user_context: dict, subject_id: int | None = None):
    if user_context["role"] == "admin":
        query = """
            SELECT id, nome, COALESCE(identificador, '') AS identificador
            FROM classes
            ORDER BY nome, identificador
        """
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    assignments = list_teacher_assignments(user_context["teacher_id"])
    if subject_id is not None:
        assignments = assignments[assignments["subject_id"] == int(subject_id)]
    if assignments.empty:
        return pd.DataFrame(columns=["id", "nome", "identificador"])

    classes_df = assignments[["class_id", "class_name", "class_identifier"]].drop_duplicates().rename(
        columns={"class_id": "id", "class_name": "nome", "class_identifier": "identificador"}
    )
    return classes_df.sort_values(["nome", "identificador"]).reset_index(drop=True)


def professor_has_assignment(teacher_id: int, subject_id: int, class_id: int) -> bool:
    query = """
        SELECT 1
        FROM teacher_subject_class
        WHERE teacher_id = :teacher_id
          AND subject_id = :subject_id
          AND class_id = :class_id
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(query),
            {"teacher_id": teacher_id, "subject_id": subject_id, "class_id": class_id},
        ).fetchone()
    return row is not None


def list_students_for_user(user_context: dict, class_id: int | None = None):
    params = {}
    query = """
        SELECT
            s.id,
            s.name,
            s.matricula,
            s.class_id,
            c.nome AS class_name,
            COALESCE(c.identificador, '') AS class_identifier
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        WHERE s.ativo = TRUE
    """
    if user_context["role"] == "professor":
        query += """
          AND s.class_id IN (
              SELECT class_id
              FROM teacher_subject_class
              WHERE teacher_id = %(teacher_id)s
          )
        """
        params["teacher_id"] = user_context["teacher_id"]

    if class_id is not None:
        query += " AND s.class_id = %(class_id)s"
        params["class_id"] = class_id

    query += " ORDER BY s.name"
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_student_lookup_for_scope(user_context: dict):
    students_df = list_students_for_user(user_context)
    lookup = {}
    for row in students_df.to_dict(orient="records"):
        lookup[row["name"]] = row
    return lookup


def create_monitoring_session(teacher_id: int, subject_id: int, class_id: int):
    now = get_local_now()
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO monitoring_sessions (
                teacher_id, subject_id, class_id, session_date, start_time, status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                teacher_id,
                subject_id,
                class_id,
                now.date(),
                now,
                SESSION_STATUS_OPEN,
            ),
        )
        session_id = cursor.fetchone()[0]
        conn.commit()
        return session_id


def close_monitoring_session(session_id: int, status: str = SESSION_STATUS_CLOSED):
    end_time = get_local_now()
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            UPDATE monitoring_sessions
            SET end_time = %s, status = %s
            WHERE id = %s
            """,
            (end_time, status, session_id),
        )
        conn.commit()


def get_monitoring_session_summary(session_id: int):
    query = """
        SELECT
            ms.id,
            ms.session_date,
            ms.start_time,
            ms.end_time,
            ms.status,
            t.id AS teacher_id,
            t.nome AS teacher_name,
            s.id AS subject_id,
            s.nome AS subject_name,
            c.id AS class_id,
            c.nome AS class_name,
            COALESCE(c.identificador, '') AS class_identifier
        FROM monitoring_sessions ms
        JOIN teachers t ON t.id = ms.teacher_id
        JOIN subjects s ON s.id = ms.subject_id
        JOIN classes c ON c.id = ms.class_id
        WHERE ms.id = :session_id
    """
    with engine.connect() as conn:
        row = conn.execute(text(query), {"session_id": session_id}).mappings().fetchone()
    return dict(row) if row else None


def insert_count_behavior(
    school,
    discipline,
    teacher,
    id_student,
    student,
    behavior,
    date,
    start_time,
    end_time,
    last_behavior=None,
):
    start_time = start_time or get_local_now().strftime("%H:%M:%S")
    end_time = end_time or get_local_now().strftime("%H:%M:%S")

    with connect_database() as (conn, cursor):
        if last_behavior is None or last_behavior != behavior:
            if last_behavior is not None:
                cursor.execute(
                    """
                    UPDATE behavior_log
                    SET end_time = %s
                    WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                    """,
                    (end_time, id_student, student, last_behavior, school, discipline, teacher),
                )

            cursor.execute(
                """
                INSERT INTO behavior_log (school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
                ON CONFLICT (student, behavior, date)
                DO UPDATE SET
                    count = behavior_log.count + 1,
                    start_time = COALESCE(behavior_log.start_time, EXCLUDED.start_time),
                    end_time = EXCLUDED.end_time
                """,
                (school, discipline, teacher, id_student, student, behavior, date, start_time, end_time),
            )
        else:
            cursor.execute(
                """
                UPDATE behavior_log
                SET end_time = %s
                WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                """,
                (end_time, id_student, student, behavior, school, discipline, teacher),
            )
        conn.commit()

    return behavior


def insert_behavior_episode(
    school,
    discipline,
    teacher,
    id_student,
    student,
    behavior,
    start_time,
    end_time,
    source="realtime",
    monitoring_session_id=None,
    student_id=None,
):
    if not student or not behavior or start_time is None or end_time is None or end_time <= start_time:
        return

    duration_seconds = float((end_time - start_time).total_seconds())
    if duration_seconds <= 0:
        return

    student_ref = student_id or id_student
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO behavior_episode (
                monitoring_session_id,
                student_id,
                school,
                discipline,
                teacher,
                id_student,
                student,
                behavior,
                start_time,
                end_time,
                duration_seconds,
                date,
                source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                monitoring_session_id,
                student_ref,
                school or DEFAULT_SCHOOL_NAME,
                discipline,
                teacher,
                student_ref,
                student,
                behavior,
                start_time,
                end_time,
                duration_seconds,
                start_time.date(),
                source,
            ),
        )
        conn.commit()


def _build_scope_conditions(user_context: dict, session_alias="ms"):
    conditions = []
    params = {}
    if user_context["role"] == "professor":
        conditions.append(f"{session_alias}.teacher_id = %(scope_teacher_id)s")
        params["scope_teacher_id"] = user_context["teacher_id"]
    return conditions, params


def _build_behavior_base_query(user_context: dict, extra_conditions=None, extra_params=None):
    conditions, params = _build_scope_conditions(user_context)
    if extra_conditions:
        conditions.extend(extra_conditions)
    if extra_params:
        params.update(extra_params)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT
            be.id,
            be.monitoring_session_id,
            be.student_id,
            be.student,
            be.behavior,
            be.start_time,
            be.end_time,
            be.duration_seconds,
            be.date,
            be.source,
            ms.session_date,
            ms.status AS session_status,
            sub.id AS subject_id,
            sub.nome AS discipline,
            cls.id AS class_id,
            cls.nome AS class_name,
            COALESCE(cls.identificador, '') AS class_identifier,
            tch.id AS teacher_id,
            tch.nome AS teacher
        FROM behavior_episode be
        JOIN monitoring_sessions ms ON ms.id = be.monitoring_session_id
        JOIN subjects sub ON sub.id = ms.subject_id
        JOIN classes cls ON cls.id = ms.class_id
        JOIN teachers tch ON tch.id = ms.teacher_id
        {where_clause}
    """
    return query, params


def fetch_behavior_dataframe(user_context: dict, filters: dict | None = None):
    filters = filters or {}
    extra_conditions = []
    params = {}

    if filters.get("student_name"):
        extra_conditions.append("be.student = %(student_name)s")
        params["student_name"] = filters["student_name"]
    if filters.get("teacher_id"):
        extra_conditions.append("ms.teacher_id = %(teacher_id)s")
        params["teacher_id"] = int(filters["teacher_id"])
    if filters.get("subject_id"):
        extra_conditions.append("ms.subject_id = %(subject_id)s")
        params["subject_id"] = int(filters["subject_id"])
    if filters.get("class_id"):
        extra_conditions.append("ms.class_id = %(class_id)s")
        params["class_id"] = int(filters["class_id"])
    if filters.get("start_date"):
        extra_conditions.append("be.date >= %(start_date)s")
        params["start_date"] = filters["start_date"]
    if filters.get("end_date"):
        extra_conditions.append("be.date <= %(end_date)s")
        params["end_date"] = filters["end_date"]

    query, base_params = _build_behavior_base_query(user_context, extra_conditions, params)
    query += " ORDER BY be.start_time ASC"
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=base_params)


def list_behavior_filter_options(user_context: dict, filters: dict | None = None):
    base_df = fetch_behavior_dataframe(user_context, filters=filters)
    if base_df.empty:
        return {
            "teachers": pd.DataFrame(columns=["id", "nome"]),
            "subjects": pd.DataFrame(columns=["id", "nome"]),
            "classes": pd.DataFrame(columns=["id", "nome", "identificador"]),
            "students": pd.DataFrame(columns=["student", "first_date", "last_date", "total_episodes"]),
        }

    teachers = (
        base_df[["teacher_id", "teacher"]]
        .drop_duplicates()
        .rename(columns={"teacher_id": "id", "teacher": "nome"})
        .sort_values("nome")
        .reset_index(drop=True)
    )
    subjects = (
        base_df[["subject_id", "discipline"]]
        .drop_duplicates()
        .rename(columns={"subject_id": "id", "discipline": "nome"})
        .sort_values("nome")
        .reset_index(drop=True)
    )
    classes_df = (
        base_df[["class_id", "class_name", "class_identifier"]]
        .drop_duplicates()
        .rename(columns={"class_id": "id", "class_name": "nome", "class_identifier": "identificador"})
        .sort_values(["nome", "identificador"])
        .reset_index(drop=True)
    )
    students = (
        base_df.groupby("student", as_index=False)
        .agg(first_date=("date", "min"), last_date=("date", "max"), total_episodes=("id", "count"))
        .sort_values("student")
        .reset_index(drop=True)
    )
    return {
        "teachers": teachers,
        "subjects": subjects,
        "classes": classes_df,
        "students": students,
    }


def df_behavior_charts(user_context: dict, filters: dict | None = None):
    df = fetch_behavior_dataframe(user_context, filters=filters)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Professor",
                "Disciplina",
                "Turma",
                "Nome do Aluno",
                "Comportamento",
                "Data",
                "Início",
                "Término",
                "Duração (s)",
                "Origem",
                "Sessão",
            ]
        )

    df["Turma"] = df["class_name"].where(df["class_identifier"].eq(""), df["class_name"] + " - " + df["class_identifier"])
    return df.rename(
        columns={
            "teacher": "Professor",
            "discipline": "Disciplina",
            "student": "Nome do Aluno",
            "behavior": "Comportamento",
            "date": "Data",
            "start_time": "Início",
            "end_time": "Término",
            "duration_seconds": "Duração (s)",
            "source": "Origem",
            "monitoring_session_id": "Sessão",
        }
    )[
        [
            "Professor",
            "Disciplina",
            "Turma",
            "Nome do Aluno",
            "Comportamento",
            "Data",
            "Início",
            "Término",
            "Duração (s)",
            "Origem",
            "Sessão",
        ]
    ]


def show_behavior_charts(user_context: dict):
    st.title("📊 Gráficos Comportamentais")
    st.caption("Análise comportamental vinculada à sessão de monitoramento, disciplina, turma e professor.")

    options = list_behavior_filter_options(user_context)
    if options["students"].empty:
        st.warning("Não há sessões monitoradas com episódios comportamentais para exibir.")
        return

    sidebar_title = "Filtros do professor" if user_context["role"] == "professor" else "Filtros globais"
    st.sidebar.header(sidebar_title)

    selected_teacher_id = None
    if user_context["role"] == "admin" and not options["teachers"].empty:
        teacher_labels = {
            int(row["id"]): row["nome"] for _, row in options["teachers"].iterrows()
        }
        teacher_choice = st.sidebar.selectbox("Professor", ["Todos"] + list(teacher_labels.values()), index=0)
        if teacher_choice != "Todos":
            selected_teacher_id = next(key for key, value in teacher_labels.items() if value == teacher_choice)

    scoped_options = list_behavior_filter_options(
        user_context,
        filters={"teacher_id": selected_teacher_id} if selected_teacher_id else None,
    )

    subject_map = {int(row["id"]): row["nome"] for _, row in scoped_options["subjects"].iterrows()}
    subject_choice = st.sidebar.selectbox("Disciplina", ["Todas"] + list(subject_map.values()), index=0)
    selected_subject_id = None
    if subject_choice != "Todas":
        selected_subject_id = next(key for key, value in subject_map.items() if value == subject_choice)

    scoped_options = list_behavior_filter_options(
        user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
        },
    )

    class_map = {
        int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
        for _, row in scoped_options["classes"].iterrows()
    }
    class_choice = st.sidebar.selectbox("Turma", ["Todas"] + list(class_map.values()), index=0)
    selected_class_id = None
    if class_choice != "Todas":
        selected_class_id = next(key for key, value in class_map.items() if value == class_choice)

    scoped_options = list_behavior_filter_options(
        user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
            "class_id": selected_class_id,
        },
    )

    student_options = scoped_options["students"]["student"].tolist()
    if not student_options:
        st.warning("Não há alunos com episódios para os filtros selecionados.")
        return
    selected_student = st.sidebar.selectbox("Aluno", student_options, index=0)

    available_dates = (
        fetch_behavior_dataframe(
            user_context,
            filters={
                "teacher_id": selected_teacher_id,
                "subject_id": selected_subject_id,
                "class_id": selected_class_id,
                "student_name": selected_student,
            },
        )["date"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    default_date = pd.to_datetime(available_dates[-1]).date() if available_dates else datetime.date.today()
    selected_date = st.sidebar.date_input("Data", value=default_date)

    df = fetch_behavior_dataframe(
        user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
            "class_id": selected_class_id,
            "student_name": selected_student,
            "start_date": selected_date,
            "end_date": selected_date,
        },
    )
    if df.empty:
        st.warning("Sem dados para os filtros selecionados.")
        return

    data_formatada = pd.to_datetime(selected_date).strftime("%d/%m/%Y")
    df_behavior = (
        df.groupby("behavior", as_index=False)["duration_seconds"]
        .sum()
        .rename(columns={"duration_seconds": "total_seconds"})
    )
    df_behavior["total_minutes"] = (df_behavior["total_seconds"] / 60.0).round(1)

    df_temporal = df[["behavior", "start_time", "end_time", "duration_seconds"]].copy()
    df_temporal["start_time"] = pd.to_datetime(df_temporal["start_time"])
    df_temporal["end_time"] = pd.to_datetime(df_temporal["end_time"])

    cores = {
        "Atento": "royalblue",
        "Perguntando": "red",
        "Escrevendo": "orange",
        "Dormindo": "purple",
        "Agitado": "green",
        "Em Pé": "gray",
        "Distraido": "brown",
        "Distraído": "brown",
    }

    title_suffix = selected_student
    fig_pie = px.pie(
        df_behavior,
        values="total_minutes",
        names="behavior",
        title=f"Distribuição do tempo por comportamento - {title_suffix} ({data_formatada})",
        hole=0.4,
        color_discrete_map=cores,
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        template="plotly_white",
    )
    fig_bar = px.bar(
        df_behavior,
        x="behavior",
        y="total_minutes",
        title=f"Tempo total por comportamento - {title_suffix} ({data_formatada})",
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        color="behavior",
        text="total_minutes",
        color_discrete_map=cores,
        template="plotly_white",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")

    fig_timeline = px.timeline(
        df_temporal,
        x_start="start_time",
        x_end="end_time",
        y="behavior",
        color="behavior",
        color_discrete_map=cores,
        title=f"Linha do tempo da aula - {title_suffix} ({data_formatada})",
        labels={"behavior": "Comportamento", "start_time": "Início", "end_time": "Fim"},
        template="plotly_white",
    )
    fig_timeline.update_yaxes(autorange="reversed")

    plotly_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pie, width="stretch", config=plotly_config)
    with col2:
        st.plotly_chart(fig_bar, width="stretch", config=plotly_config)
    st.plotly_chart(fig_timeline, width="stretch", config=plotly_config)

    export_df = df_behavior_charts(
        user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
            "class_id": selected_class_id,
            "student_name": selected_student,
            "start_date": selected_date,
            "end_date": selected_date,
        },
    )
    st.download_button(
        "Baixar base filtrada (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"graficos_{selected_student}_{selected_date}.csv",
        mime="text/csv",
    )

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)

    def add_plot_page(fig, titulo):
        temp_buf = io.BytesIO()
        try:
            fig.write_image(temp_buf, format="png")
        except Exception:
            return
        temp_buf.seek(0)
        image = Image.open(temp_buf)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.convert("RGB").save(temp_file.name)
            temp_path = temp_file.name
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 800, titulo)
        c.drawImage(temp_path, 40, 120, width=500, preserveAspectRatio=True, mask="auto")
        c.showPage()
        os.unlink(temp_path)

    add_plot_page(fig_pie, "Distribuição do tempo por comportamento")
    add_plot_page(fig_bar, "Tempo total por comportamento")
    add_plot_page(fig_timeline, "Linha do tempo da aula")
    c.save()
    pdf_buf.seek(0)
    st.download_button(
        "Baixar gráficos em PDF",
        data=pdf_buf,
        file_name=f"graficos_{selected_student}_{selected_date}.pdf",
        mime="application/pdf",
    )
