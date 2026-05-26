from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.monitoring import (
    MonitoringClass,
    MonitoringOptionsResponse,
    MonitoringSessionCreate,
    MonitoringSessionRead,
    MonitoringSessionSummary,
    MonitoringSubject,
)
from app.schemas.users import CurrentUser


LESSON_TYPES = ["Exposição", "Atividade", "Prova", "Revisão", "Outro"]


def list_monitoring_options(db: Session, current_user: CurrentUser) -> MonitoringOptionsResponse:
    params: dict[str, object] = {}
    if current_user.role == "admin":
        subjects_query = """
            SELECT id, nome
            FROM subjects
            ORDER BY nome
        """
        classes_query = """
            SELECT id, nome, COALESCE(identificador, '') AS identificador
            FROM classes
            ORDER BY nome, identificador
        """
    else:
        params["teacher_id"] = current_user.teacher_id
        subjects_query = """
            SELECT DISTINCT s.id, s.nome
            FROM subjects s
            JOIN teacher_subject_class tsc ON tsc.subject_id = s.id
            WHERE tsc.teacher_id = :teacher_id
            ORDER BY s.nome
        """
        classes_query = """
            SELECT DISTINCT c.id, c.nome, COALESCE(c.identificador, '') AS identificador
            FROM classes c
            JOIN teacher_subject_class tsc ON tsc.class_id = c.id
            WHERE tsc.teacher_id = :teacher_id
            ORDER BY c.nome, identificador
        """

    subject_rows = db.execute(text(subjects_query), params).mappings().all()
    class_rows = db.execute(text(classes_query), params).mappings().all()
    return MonitoringOptionsResponse(
        subjects=[MonitoringSubject(**dict(row)) for row in subject_rows],
        classes=[MonitoringClass(**dict(row)) for row in class_rows],
        lesson_types=LESSON_TYPES,
    )


def ensure_teacher_can_monitor(db: Session, current_user: CurrentUser, subject_id: int, class_id: int) -> None:
    if current_user.role != "professor":
        return

    assignment = db.execute(
        text(
            """
            SELECT 1
            FROM teacher_subject_class
            WHERE teacher_id = :teacher_id
              AND subject_id = :subject_id
              AND class_id = :class_id
            LIMIT 1
            """
        ),
        {
            "teacher_id": current_user.teacher_id,
            "subject_id": subject_id,
            "class_id": class_id,
        },
    ).first()
    if assignment is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teacher assignment not found")


def create_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    payload: MonitoringSessionCreate,
) -> MonitoringSessionRead:
    ensure_teacher_can_monitor(db, current_user, payload.subject_id, payload.class_id)
    teacher_id = current_user.teacher_id
    if teacher_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User has no teacher profile")

    row = db.execute(
        text(
            """
            INSERT INTO monitoring_sessions (
                teacher_id, subject_id, class_id, lesson_type, session_date, start_time, status
            )
            VALUES (
                :teacher_id, :subject_id, :class_id, :lesson_type, CURRENT_DATE, CURRENT_TIMESTAMP, 'em_andamento'
            )
            RETURNING id, status
            """
        ),
        {
            "teacher_id": teacher_id,
            "subject_id": payload.subject_id,
            "class_id": payload.class_id,
            "lesson_type": payload.lesson_type,
        },
    ).mappings().one()
    db.commit()
    return MonitoringSessionRead(id=row["id"], status=row["status"])


def close_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    session_id: int,
    close_status: str,
) -> MonitoringSessionRead:
    conditions = ["id = :session_id"]
    params: dict[str, object] = {"session_id": session_id, "status": close_status}
    if current_user.role == "professor":
        conditions.append("teacher_id = :teacher_id")
        params["teacher_id"] = current_user.teacher_id

    row = db.execute(
        text(
            f"""
            UPDATE monitoring_sessions
            SET end_time = CURRENT_TIMESTAMP, status = :status
            WHERE {" AND ".join(conditions)}
            RETURNING id, status
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitoring session not found")
    db.commit()
    return MonitoringSessionRead(id=row["id"], status=row["status"])


def get_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    session_id: int,
) -> MonitoringSessionSummary:
    conditions = ["id = :session_id"]
    params: dict[str, object] = {"session_id": session_id}
    if current_user.role == "professor":
        conditions.append("teacher_id = :teacher_id")
        params["teacher_id"] = current_user.teacher_id

    row = db.execute(
        text(
            f"""
            SELECT id, status, start_time, end_time, lesson_type
            FROM monitoring_sessions
            WHERE {" AND ".join(conditions)}
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitoring session not found")
    return MonitoringSessionSummary(**dict(row))
