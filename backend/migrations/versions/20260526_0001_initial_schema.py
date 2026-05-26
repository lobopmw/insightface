"""initial schema compatible with legacy Streamlit app

Revision ID: 20260526_0001
Revises:
Create Date: 2026-05-26
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260526_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
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
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255)")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'professor'")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS ativo BOOLEAN NOT NULL DEFAULT TRUE")
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            nome VARCHAR(255) NOT NULL
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id SERIAL PRIMARY KEY,
            nome VARCHAR(255) NOT NULL UNIQUE
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            nome VARCHAR(255) NOT NULL,
            identificador VARCHAR(255),
            CONSTRAINT uq_classes_nome_identificador UNIQUE (nome, identificador)
        )
        """
    )

    op.execute(
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

    op.execute(
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
    op.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS matricula VARCHAR(255)")
    op.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS class_id INTEGER REFERENCES classes(id) ON DELETE SET NULL")
    op.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS ativo BOOLEAN NOT NULL DEFAULT TRUE")
    op.execute(
        "ALTER TABLE students ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )

    op.execute(
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

    op.execute(
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
    op.execute("ALTER TABLE monitoring_sessions ADD COLUMN IF NOT EXISTS lesson_type VARCHAR(50)")

    op.execute(
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

    op.execute(
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
    op.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS monitoring_session_id INTEGER REFERENCES monitoring_sessions(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS student_id VARCHAR(255) REFERENCES students(id) ON DELETE SET NULL"
    )
    op.execute("ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS lesson_type VARCHAR(50)")
    op.execute(
        "ALTER TABLE behavior_episode ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP"
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_session_student
        ON behavior_episode (monitoring_session_id, student_id, start_time)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_date
        ON behavior_episode (student, date)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_start_time
        ON behavior_episode (student, start_time)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_monitoring_sessions_scope
        ON monitoring_sessions (teacher_id, subject_id, class_id, session_date)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_students_class_matricula_normalized
        ON students (class_id, UPPER(BTRIM(matricula)))
        WHERE NULLIF(BTRIM(COALESCE(matricula, '')), '') IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_students_class_matricula_normalized")
    op.execute("DROP INDEX IF EXISTS idx_monitoring_sessions_scope")
    op.execute("DROP INDEX IF EXISTS idx_behavior_episode_student_start_time")
    op.execute("DROP INDEX IF EXISTS idx_behavior_episode_student_date")
    op.execute("DROP INDEX IF EXISTS idx_behavior_episode_session_student")
    op.execute("DROP TABLE IF EXISTS behavior_episode")
    op.execute("DROP TABLE IF EXISTS behavior_log")
    op.execute("DROP TABLE IF EXISTS monitoring_sessions")
    op.execute("DROP TABLE IF EXISTS face_embeddings")
    op.execute("DROP TABLE IF EXISTS students")
    op.execute("DROP TABLE IF EXISTS teacher_subject_class")
    op.execute("DROP TABLE IF EXISTS classes")
    op.execute("DROP TABLE IF EXISTS subjects")
    op.execute("DROP TABLE IF EXISTS teachers")
    op.execute("DROP TABLE IF EXISTS users")
