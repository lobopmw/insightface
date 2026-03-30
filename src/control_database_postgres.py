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
DEFAULT_LESSON_TYPE = "Exposição"
UNKNOWN_LESSON_TYPE = "Não informado"


def get_local_now() -> datetime.datetime:
    return datetime.datetime.now(ZoneInfo(APP_TIMEZONE)).replace(tzinfo=None)


def _chart_icon_svg(kind: str) -> str:
    icons = {
        "summary": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="4" y="12" width="3" height="7" rx="1.2" fill="currentColor"></rect>
                <rect x="10.5" y="8" width="3" height="11" rx="1.2" fill="currentColor" opacity="0.92"></rect>
                <rect x="17" y="5" width="3" height="14" rx="1.2" fill="currentColor" opacity="0.82"></rect>
                <path d="M4 20h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" opacity="0.65"></path>
            </svg>
        """,
        "pie": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 3a9 9 0 1 0 9 9h-9V3Z" fill="currentColor"></path>
                <path d="M14 3.4A8.6 8.6 0 0 1 20.6 10H14V3.4Z" fill="currentColor" opacity="0.55"></path>
            </svg>
        """,
        "bar": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="4" y="11" width="3.2" height="8" rx="1.1" fill="currentColor"></rect>
                <rect x="10.4" y="7.5" width="3.2" height="11.5" rx="1.1" fill="currentColor" opacity="0.9"></rect>
                <rect x="16.8" y="4.5" width="3.2" height="14.5" rx="1.1" fill="currentColor" opacity="0.8"></rect>
            </svg>
        """,
        "time": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.8" fill="none"></circle>
                <path d="M12 7.7v4.6l3.1 1.9" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
            </svg>
        """,
        "empty": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.8" fill="none"></circle>
                <path d="M12 8v4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"></path>
                <circle cx="12" cy="16.8" r="1" fill="currentColor"></circle>
            </svg>
        """,
    }
    return icons.get(kind, icons["summary"])


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
            lesson_type VARCHAR(50),
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
            lesson_type VARCHAR(50),
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
        "ALTER TABLE monitoring_sessions ADD COLUMN IF NOT EXISTS lesson_type VARCHAR(50)"
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS monitoring_session_id INTEGER REFERENCES monitoring_sessions(id) ON DELETE CASCADE"
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS student_id VARCHAR(255) REFERENCES students(id) ON DELETE SET NULL"
    )
    cursor.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS lesson_type VARCHAR(50)"
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


def create_monitoring_session(
    teacher_id: int,
    subject_id: int,
    class_id: int,
    lesson_type: str = DEFAULT_LESSON_TYPE,
):
    now = get_local_now()
    with connect_database() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO monitoring_sessions (
                teacher_id, subject_id, class_id, lesson_type, session_date, start_time, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                teacher_id,
                subject_id,
                class_id,
                lesson_type or DEFAULT_LESSON_TYPE,
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
            COALESCE(ms.lesson_type, :unknown_lesson_type) AS lesson_type,
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
        row = conn.execute(
            text(query),
            {"session_id": session_id, "unknown_lesson_type": UNKNOWN_LESSON_TYPE},
        ).mappings().fetchone()
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
    lesson_type=None,
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
                lesson_type,
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                monitoring_session_id,
                student_ref,
                lesson_type,
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
            COALESCE(be.lesson_type, ms.lesson_type, %(unknown_lesson_type)s) AS lesson_type,
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
    params = {"unknown_lesson_type": UNKNOWN_LESSON_TYPE}

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
                "Tipo de aula",
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
            "lesson_type": "Tipo de aula",
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
            "Tipo de aula",
        ]
    ]


def _format_duration_label(total_seconds: float) -> str:
    if total_seconds >= 60:
        return f"{total_seconds / 60.0:.1f} min"
    return f"{total_seconds:.0f} s"


def _build_behavior_dashboard_summary(
    df_behavior: pd.DataFrame,
    df_context_share: pd.DataFrame,
) -> list[str]:
    if df_behavior.empty:
        return ["Ainda não há dados suficientes para gerar um resumo textual."]

    summary_lines: list[str] = []
    top_behavior = df_behavior.sort_values("share_percentage", ascending=False).iloc[0]
    summary_lines.append(
        f"O aluno permaneceu a maior parte do tempo no comportamento {top_behavior['behavior']} "
        f"({top_behavior['share_percentage']:.1f}% do tempo monitorado)."
    )

    non_attentive = df_behavior[
        df_behavior["behavior"].astype(str).str.lower().isin(["distraído", "distraido", "dormindo", "agitado"])
    ].sort_values("share_percentage", ascending=False)
    if not non_attentive.empty:
        main_non_attentive = non_attentive.iloc[0]
        summary_lines.append(
            f"O comportamento {main_non_attentive['behavior']} acumulou "
            f"{_format_duration_label(float(main_non_attentive['total_seconds']))} no recorte selecionado."
        )

    if not df_context_share.empty and df_context_share["lesson_type"].nunique() >= 2:
        context_top = (
            df_context_share.sort_values(["share_percentage", "total_seconds"], ascending=[False, False])
            .iloc[0]
        )
        summary_lines.append(
            f"Foi observada maior presença de {context_top['behavior']} em aulas do tipo "
            f"{context_top['lesson_type']} ({context_top['share_percentage']:.1f}% do tempo nesse contexto)."
        )

        attentive_by_context = df_context_share[
            df_context_share["behavior"].astype(str).str.lower().eq("atento")
        ].sort_values("share_percentage", ascending=False)
        distracted_by_context = df_context_share[
            df_context_share["behavior"].astype(str).str.lower().isin(["distraído", "distraido"])
        ].sort_values("share_percentage", ascending=False)
        if not attentive_by_context.empty and not distracted_by_context.empty:
            best_attention = attentive_by_context.iloc[0]
            highest_distraction = distracted_by_context.iloc[0]
            if best_attention["lesson_type"] != highest_distraction["lesson_type"]:
                summary_lines.append(
                    f"Em {best_attention['lesson_type']}, o nível de atenção foi superior ao observado em "
                    f"{highest_distraction['lesson_type']}."
                )

    return summary_lines[:3]


def show_behavior_charts(user_context: dict):
    st.markdown(
        """
        <style>
        .charts-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.1rem 0 1.1rem 0;
        }
        .charts-hero-main {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .charts-hero-icon {
            width: 66px;
            height: 66px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            background: linear-gradient(180deg, rgba(103,83,255,0.28) 0%, rgba(69,55,166,0.2) 100%);
            border: 1px solid rgba(121,94,255,0.25);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .charts-title {
            margin: 0;
            color: #F4F7FB;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .charts-subtitle {
            margin: 0.35rem 0 0 0;
            color: #A6AFBE;
            font-size: 1rem;
            line-height: 1.55;
        }
        .charts-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            border-radius: 18px;
            border: 1px solid rgba(107,93,255,0.34);
            background: linear-gradient(180deg, rgba(36,40,68,0.96) 0%, rgba(26,29,50,0.98) 100%);
            color: #DCE3F4;
            padding: 0.95rem 1.1rem;
            font-size: 0.98rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .charts-chip-dot {
            color: #7B69FF;
            font-size: 0.7rem;
        }
        .charts-section-card {
            border: 1px solid rgba(107,93,255,0.22);
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(20,24,36,0.96) 0%, rgba(16,20,30,0.99) 100%);
            box-shadow: 0 16px 34px rgba(0,0,0,0.16);
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .charts-section-head {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 1rem 1.15rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .charts-section-icon {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            color: #F5F7FB;
            background: linear-gradient(180deg, #6F5BFF 0%, #4D38D2 100%);
        }
        .charts-section-icon svg,
        .charts-card-mini-icon svg {
            width: 18px;
            height: 18px;
            display: block;
        }
        .charts-section-title {
            color: #F4F7FB;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0;
        }
        .charts-card-mini-title {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: #F4F7FB;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .charts-card-mini-icon {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, rgba(108,93,255,0.22) 0%, rgba(76,60,203,0.18) 100%);
            color: #8E7BFF;
            font-size: 0.92rem;
            flex: 0 0 auto;
        }
        .charts-card-mini-subtitle {
            color: #97A2B5;
            margin-bottom: 0.35rem;
        }
        .charts-panel-body {
            padding: 1rem 1rem 0.8rem 1rem;
        }
        .charts-info-note {
            border-radius: 18px;
            border: 1px solid rgba(69,107,214,0.22);
            background: linear-gradient(180deg, rgba(23,48,92,0.46) 0%, rgba(18,38,72,0.34) 100%);
            color: #9CC7FF;
            padding: 1rem 1.1rem;
            line-height: 1.5;
            margin-top: 1rem;
        }
        .charts-filter-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(20,24,36,0.96) 0%, rgba(16,20,30,0.99) 100%);
            box-shadow: 0 18px 36px rgba(0,0,0,0.18);
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .charts-filter-title-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.65rem;
        }
        .charts-filter-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, #7B63FF 0%, #563DDF 100%);
            color: white;
            font-size: 1.1rem;
        }
        .charts-filter-heading {
            color: #F4F7FB;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0;
        }
        .charts-filter-subtitle {
            color: #9AA4B6;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0 0 0.8rem 0;
        }
        .charts-filter-divider {
            height: 1px;
            background: rgba(255,255,255,0.08);
            margin: 0.8rem 0 1rem 0;
        }
        .charts-applied-box {
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            background: rgba(255,255,255,0.02);
            padding: 1rem;
        }
        .charts-applied-title {
            color: #DDF8E9;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.85rem;
        }
        .charts-applied-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.5rem;
            color: #D4D9E5;
            font-size: 0.95rem;
        }
        .charts-applied-row span:first-child {
            color: #97A2B5;
        }
        .charts-tip-card {
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(56,40,111,0.4) 0%, rgba(33,24,67,0.32) 100%);
            border: 1px solid rgba(124,95,255,0.14);
            padding: 1rem;
            color: #D6DBE8;
            line-height: 1.55;
        }
        .charts-tip-title {
            font-weight: 800;
            color: #F4F7FB;
            margin-bottom: 0.35rem;
        }
        @media (max-width: 1100px) {
            .charts-hero {
                flex-direction: column;
                align-items: flex-start;
            }
            .charts-chip {
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    options = list_behavior_filter_options(user_context)
    filter_title = "Filtros do professor" if user_context["role"] == "professor" else "Filtros globais"
    state_prefix = "behavior_chart_filters"
    selected_teacher_id = st.session_state.get(f"{state_prefix}_teacher_id")
    selected_subject_id = st.session_state.get(f"{state_prefix}_subject_id")
    selected_class_id = st.session_state.get(f"{state_prefix}_class_id")
    selected_student_state = st.session_state.get(f"{state_prefix}_student")
    selected_date_state = st.session_state.get(f"{state_prefix}_date")

    left_col, right_col = st.columns([2.55, 1], gap="large")

    def _render_charts_empty_state(title: str, message: str, student_label: str | None = None, date_label: str | None = None):
        fallback_student = student_label or "Selecione um aluno"
        fallback_date = date_label or pd.to_datetime(datetime.date.today()).strftime("%d/%m/%Y")
        with left_col:
            st.markdown(
                f"""
                <div class="charts-hero">
                    <div class="charts-hero-main">
                        <div class="charts-hero-icon">📊</div>
                        <div>
                            <h1 class="charts-title">Gráficos Comportamentais</h1>
                            <p class="charts-subtitle">Análise comportamental vinculada à sessão de monitoramento, disciplina, turma e professor.</p>
                        </div>
                    </div>
                    <div class="charts-chip">
                        <span>📅</span>
                        <span>Análise para:</span>
                        <strong>{fallback_student}</strong>
                        <span class="charts-chip-dot">●</span>
                        <span>{fallback_date}</span>
                    </div>
                </div>
                <div class="charts-section-card">
                    <div class="charts-section-head">
                        <div class="charts-section-icon">{_chart_icon_svg("empty")}</div>
                        <div class="charts-section-title">{title}</div>
                    </div>
                    <div class="charts-panel-body">
                        <div class="charts-info-note" style="margin-top:0;">
                            {message}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if options["students"].empty:
        _render_charts_empty_state(
            "Sem dados disponíveis",
            "Ainda não há sessões monitoradas com episódios comportamentais suficientes para gerar gráficos.",
        )
        return

    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="charts-filter-title-row">
                    <div class="charts-filter-icon">⏷</div>
                    <div class="charts-filter-heading">{filter_title}</div>
                </div>
                <div class="charts-filter-subtitle">Refine os gráficos por disciplina, turma, aluno e data.</div>
                <div class="charts-filter-divider"></div>
                """,
                unsafe_allow_html=True,
            )

            teacher_choice = "Todos"
            if user_context["role"] == "admin" and not options["teachers"].empty:
                teacher_labels = {
                    int(row["id"]): row["nome"] for _, row in options["teachers"].iterrows()
                }
                teacher_values = ["Todos"] + list(teacher_labels.values())
                teacher_default = "Todos"
                if selected_teacher_id in teacher_labels:
                    teacher_default = teacher_labels[selected_teacher_id]
                teacher_choice = st.selectbox(
                    "Professor",
                    teacher_values,
                    index=teacher_values.index(teacher_default),
                    key=f"{state_prefix}_teacher_choice",
                )
                selected_teacher_id = None if teacher_choice == "Todos" else next(
                    key for key, value in teacher_labels.items() if value == teacher_choice
                )
            else:
                selected_teacher_id = None

            scoped_options = list_behavior_filter_options(
                user_context,
                filters={"teacher_id": selected_teacher_id} if selected_teacher_id else None,
            )

            subject_map = {int(row["id"]): row["nome"] for _, row in scoped_options["subjects"].iterrows()}
            subject_values = ["Todas"] + list(subject_map.values())
            subject_default = "Todas"
            if selected_subject_id in subject_map:
                subject_default = subject_map[selected_subject_id]
            subject_choice = st.selectbox(
                "Disciplina",
                subject_values,
                index=subject_values.index(subject_default),
                key=f"{state_prefix}_subject_choice",
            )
            selected_subject_id = None if subject_choice == "Todas" else next(
                key for key, value in subject_map.items() if value == subject_choice
            )

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
            class_values = ["Todas"] + list(class_map.values())
            class_default = "Todas"
            if selected_class_id in class_map:
                class_default = class_map[selected_class_id]
            class_choice = st.selectbox(
                "Turma",
                class_values,
                index=class_values.index(class_default),
                key=f"{state_prefix}_class_choice",
            )
            selected_class_id = None if class_choice == "Todas" else next(
                key for key, value in class_map.items() if value == class_choice
            )

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
                _render_charts_empty_state(
                    "Nenhum aluno encontrado",
                    "Os filtros atuais não retornaram alunos com episódios comportamentais registrados. Ajuste a disciplina, turma ou data.",
                    selected_student_state,
                    pd.to_datetime(selected_date_state).strftime("%d/%m/%Y") if selected_date_state else None,
                )
                return
            if selected_student_state not in student_options:
                selected_student_state = student_options[0]
            selected_student = st.selectbox(
                "Aluno",
                student_options,
                index=student_options.index(selected_student_state),
                key=f"{state_prefix}_student_choice",
            )

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
            if selected_date_state is not None:
                try:
                    selected_date_candidate = pd.to_datetime(selected_date_state).date()
                except Exception:
                    selected_date_candidate = default_date
            else:
                selected_date_candidate = default_date
            selected_date = st.date_input(
                "Data",
                value=selected_date_candidate,
                key=f"{state_prefix}_date_choice",
            )

            button_col1, button_col2 = st.columns(2, gap="small")
            with button_col1:
                apply_filters = st.button("Aplicar Filtros", type="primary", use_container_width=True)
            with button_col2:
                clear_filters = st.button("Limpar Filtros", use_container_width=True)

            if clear_filters:
                for key in (
                    f"{state_prefix}_teacher_id",
                    f"{state_prefix}_subject_id",
                    f"{state_prefix}_class_id",
                    f"{state_prefix}_student",
                    f"{state_prefix}_date",
                    f"{state_prefix}_teacher_choice",
                    f"{state_prefix}_subject_choice",
                    f"{state_prefix}_class_choice",
                    f"{state_prefix}_student_choice",
                    f"{state_prefix}_date_choice",
                ):
                    st.session_state.pop(key, None)
                st.rerun()

            if apply_filters or f"{state_prefix}_student" not in st.session_state:
                st.session_state[f"{state_prefix}_teacher_id"] = selected_teacher_id
                st.session_state[f"{state_prefix}_subject_id"] = selected_subject_id
                st.session_state[f"{state_prefix}_class_id"] = selected_class_id
                st.session_state[f"{state_prefix}_student"] = selected_student
                st.session_state[f"{state_prefix}_date"] = selected_date.isoformat()

            selected_teacher_id = st.session_state.get(f"{state_prefix}_teacher_id")
            selected_subject_id = st.session_state.get(f"{state_prefix}_subject_id")
            selected_class_id = st.session_state.get(f"{state_prefix}_class_id")
            selected_student = st.session_state.get(f"{state_prefix}_student", selected_student)
            selected_date = pd.to_datetime(st.session_state.get(f"{state_prefix}_date", selected_date.isoformat())).date()

            applied_subject = subject_choice if selected_subject_id else "Todas"
            applied_class = class_choice if selected_class_id else "Todas"

            st.markdown(
                f"""
                <div class="charts-filter-divider"></div>
                <div class="charts-applied-box">
                    <div class="charts-applied-title">Filtros aplicados</div>
                    <div class="charts-applied-row"><span>Disciplina:</span><span>{applied_subject}</span></div>
                    <div class="charts-applied-row"><span>Turma:</span><span>{applied_class}</span></div>
                    <div class="charts-applied-row"><span>Aluno:</span><span>{selected_student}</span></div>
                    <div class="charts-applied-row"><span>Data:</span><span>{pd.to_datetime(selected_date).strftime("%d/%m/%Y")}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="charts-tip-card">
                <div class="charts-tip-title">Dica</div>
                Selecione os filtros desejados e clique em <strong>Aplicar Filtros</strong> para atualizar os gráficos.
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        _render_charts_empty_state(
            "Nenhum registro no período",
            "Não foram encontrados episódios comportamentais para os filtros aplicados. Tente alterar a data ou ampliar o escopo da disciplina e da turma.",
            selected_student,
            pd.to_datetime(selected_date).strftime("%d/%m/%Y"),
        )
        return

    data_formatada = pd.to_datetime(selected_date).strftime("%d/%m/%Y")
    df_behavior = (
        df.groupby("behavior", as_index=False)["duration_seconds"]
        .sum()
        .rename(columns={"duration_seconds": "total_seconds"})
    )
    df_behavior["total_minutes"] = (df_behavior["total_seconds"] / 60.0).round(1)
    total_monitored_seconds = float(df_behavior["total_seconds"].sum())
    df_behavior["share_percentage"] = (
        (df_behavior["total_seconds"] / total_monitored_seconds) * 100.0 if total_monitored_seconds > 0 else 0.0
    ).round(1)
    df_behavior["duration_label"] = df_behavior["total_seconds"].apply(_format_duration_label)

    df_temporal = df[["behavior", "start_time", "end_time", "duration_seconds"]].copy()
    df_temporal["start_time"] = pd.to_datetime(df_temporal["start_time"])
    df_temporal["end_time"] = pd.to_datetime(df_temporal["end_time"])

    df_context = fetch_behavior_dataframe(
        user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
            "class_id": selected_class_id,
            "student_name": selected_student,
        },
    )
    df_context_share = pd.DataFrame(columns=["lesson_type", "behavior", "total_seconds", "share_percentage"])
    if not df_context.empty:
        df_context_share = (
            df_context.groupby(["lesson_type", "behavior"], as_index=False)["duration_seconds"]
            .sum()
            .rename(columns={"duration_seconds": "total_seconds"})
        )
        lesson_totals = (
            df_context_share.groupby("lesson_type", as_index=False)["total_seconds"]
            .sum()
            .rename(columns={"total_seconds": "lesson_total_seconds"})
        )
        df_context_share = df_context_share.merge(lesson_totals, on="lesson_type", how="left")
        df_context_share["share_percentage"] = (
            (df_context_share["total_seconds"] / df_context_share["lesson_total_seconds"]) * 100.0
        ).round(1)

    summary_lines = _build_behavior_dashboard_summary(df_behavior, df_context_share)

    cores = {
        "Atento": "#5C6CFF",
        "Perguntando": "#FF7B6B",
        "Escrevendo": "#F3A54A",
        "Dormindo": "#8B5CF6",
        "Agitado": "#2EC27E",
        "Em Pé": "#8E97A9",
        "Distraido": "#C17D48",
        "Distraído": "#C17D48",
    }
    title_suffix = selected_student
    fig_pie = px.pie(
        df_behavior,
        values="total_minutes",
        names="behavior",
        hole=0.4,
        color_discrete_map=cores,
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        template="plotly_dark",
    )
    fig_pie.update_layout(
        title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE3F4"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02, xanchor="left"),
        uniformtext_minsize=18,
        uniformtext_mode="show",
    )
    fig_pie.update_traces(
        texttemplate="<b>%{percent}</b>",
        textposition="inside",
        insidetextfont=dict(size=24, color="#F7FAFF"),
    )
    fig_bar = px.bar(
        df_behavior,
        x="behavior",
        y="total_minutes",
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        color="behavior",
        text="total_minutes",
        color_discrete_map=cores,
        template="plotly_dark",
    )
    fig_bar.update_traces(
        texttemplate="<b>%{text:.1f}</b>",
        textposition="outside",
        textfont=dict(size=20, color="#F7FAFF"),
        cliponaxis=False,
    )
    fig_bar.update_layout(
        title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE3F4"),
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        showlegend=False,
    )

    fig_percent = px.bar(
        df_behavior.sort_values("share_percentage", ascending=False),
        x="behavior",
        y="share_percentage",
        labels={"behavior": "Comportamento", "share_percentage": "Percentual do tempo (%)"},
        color="behavior",
        text="share_percentage",
        color_discrete_map=cores,
        template="plotly_dark",
    )
    fig_percent.update_traces(
        texttemplate="<b>%{text:.1f}%</b>",
        textposition="outside",
        textfont=dict(size=18, color="#F7FAFF"),
        cliponaxis=False,
    )
    fig_percent.update_layout(
        title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE3F4"),
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        showlegend=False,
    )

    fig_timeline = px.timeline(
        df_temporal,
        x_start="start_time",
        x_end="end_time",
        y="behavior",
        color="behavior",
        color_discrete_map=cores,
        labels={"behavior": "Comportamento", "start_time": "Início", "end_time": "Fim"},
        template="plotly_dark",
    )
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(
        title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE3F4"),
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend_title_text="Comportamento",
    )

    fig_context = None
    if not df_context_share.empty and df_context_share["lesson_type"].nunique() >= 1:
        fig_context = px.bar(
            df_context_share.sort_values(["lesson_type", "share_percentage"], ascending=[True, False]),
            x="lesson_type",
            y="share_percentage",
            color="behavior",
            text="share_percentage",
            barmode="stack",
            color_discrete_map=cores,
            labels={
                "lesson_type": "Tipo de aula",
                "share_percentage": "Percentual do tempo (%)",
                "behavior": "Comportamento",
            },
            template="plotly_dark",
        )
        fig_context.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=14, color="#F7FAFF"),
            cliponaxis=False,
        )
        fig_context.update_layout(
            title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE3F4"),
            margin=dict(l=10, r=10, t=10, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", ticksuffix="%"),
            legend_title_text="Comportamento",
        )

    plotly_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    with left_col:
        st.markdown(
            f"""
            <div class="charts-hero">
                <div class="charts-hero-main">
                    <div class="charts-hero-icon">📊</div>
                    <div>
                        <h1 class="charts-title">Gráficos Comportamentais</h1>
                        <p class="charts-subtitle">Análise comportamental vinculada à sessão de monitoramento, disciplina, turma e professor.</p>
                    </div>
                </div>
                <div class="charts-chip">
                    <span>📅</span>
                    <span>Análise para:</span>
                    <strong>{selected_student}</strong>
                    <span class="charts-chip-dot">●</span>
                    <span>{data_formatada}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_tab, timeline_tab, lesson_type_tab = st.tabs(
            ["Resumo", "Linha do tempo", "Tipo de aula"]
        )

        with summary_tab:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="charts-section-head" style="margin:-1rem -1rem 1rem -1rem;">
                        <div class="charts-section-icon">{_chart_icon_svg("summary")}</div>
                        <div class="charts-section-title">Resumo dos Comportamentos</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="charts-card-mini-title">
                                <span class="charts-card-mini-icon">{_chart_icon_svg("pie")}</span>
                                <span>Distribuição do tempo por comportamento</span>
                            </div>
                            <div class="charts-card-mini-subtitle">{selected_student} • {data_formatada}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(fig_pie, width="stretch", config=plotly_config)
                with col2:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="charts-card-mini-title">
                                <span class="charts-card-mini-icon">{_chart_icon_svg("bar")}</span>
                                <span>Tempo total por comportamento</span>
                            </div>
                            <div class="charts-card-mini-subtitle">{selected_student} • {data_formatada}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(fig_bar, width="stretch", config=plotly_config)

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="charts-card-mini-title">
                            <span class="charts-card-mini-icon">{_chart_icon_svg("bar")}</span>
                            <span>Percentual do tempo por comportamento</span>
                        </div>
                        <div class="charts-card-mini-subtitle">{selected_student} • {data_formatada}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(fig_percent, width="stretch", config=plotly_config)

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="charts-card-mini-title">
                            <span class="charts-card-mini-icon">{_chart_icon_svg("summary")}</span>
                            <span>Resumo interpretável</span>
                        </div>
                        <div class="charts-card-mini-subtitle">{selected_student} • leitura objetiva dos dados</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for line in summary_lines:
                        st.markdown(f"- {line}")

        with timeline_tab:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="charts-section-head" style="margin:-1rem -1rem 1rem -1rem;">
                        <div class="charts-section-icon">{_chart_icon_svg("time")}</div>
                        <div class="charts-section-title">Linha do tempo da aula</div>
                    </div>
                    <div style='color:#97A2B5; margin-bottom:0.4rem;'>{selected_student} • {data_formatada}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_timeline, width="stretch", config=plotly_config)

        with lesson_type_tab:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="charts-section-head" style="margin:-1rem -1rem 1rem -1rem;">
                        <div class="charts-section-icon">{}</div>
                        <div class="charts-section-title">Comparação por tipo de aula</div>
                    </div>
                    """.format(_chart_icon_svg("summary")),
                    unsafe_allow_html=True,
                )
                if fig_context is not None and df_context_share["lesson_type"].nunique() >= 2:
                    st.plotly_chart(fig_context, width="stretch", config=plotly_config)
                elif fig_context is not None:
                    st.plotly_chart(fig_context, width="stretch", config=plotly_config)
                    st.caption("Ainda há apenas um tipo de aula com registros para este aluno no escopo atual.")
                else:
                    st.info("Ainda não há dados suficientes para comparar comportamentos entre tipos de aula.")

        st.markdown(
            """
            <div class="charts-info-note">
                Os gráficos apresentam os comportamentos registrados no período selecionado e, quando houver histórico suficiente, uma comparação adicional por tipo de aula.
            </div>
            """,
            unsafe_allow_html=True,
        )
