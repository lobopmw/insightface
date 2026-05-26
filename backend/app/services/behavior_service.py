from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session


DEFAULT_SCHOOL_NAME = "Escola Estadual Criança Esperança"


def insert_behavior_episode(
    db: Session,
    school: str | None,
    discipline: str | None,
    teacher: str | None,
    id_student: str | None,
    student: str,
    behavior: str,
    start_time: datetime,
    end_time: datetime,
    source: str = "realtime",
    monitoring_session_id: int | None = None,
    student_id: str | None = None,
    lesson_type: str | None = None,
) -> None:
    if not student or not behavior or end_time <= start_time:
        return

    duration_seconds = float((end_time - start_time).total_seconds())
    if duration_seconds <= 0:
        return

    student_ref = student_id or id_student
    db.execute(
        text(
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
            VALUES (
                :monitoring_session_id,
                :student_id,
                :lesson_type,
                :school,
                :discipline,
                :teacher,
                :id_student,
                :student,
                :behavior,
                :start_time,
                :end_time,
                :duration_seconds,
                :date,
                :source
            )
            """
        ),
        {
            "monitoring_session_id": monitoring_session_id,
            "student_id": student_ref,
            "lesson_type": lesson_type,
            "school": school or DEFAULT_SCHOOL_NAME,
            "discipline": discipline,
            "teacher": teacher,
            "id_student": student_ref,
            "student": student,
            "behavior": behavior,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "date": start_time.date(),
            "source": source,
        },
    )
    db.commit()
