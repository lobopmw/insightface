from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.reports import BehaviorEpisodeRead, ReportSummaryItem
from app.schemas.users import CurrentUser


def get_behavior_summary(db: Session, current_user: CurrentUser) -> list[ReportSummaryItem]:
    params: dict[str, object] = {}
    teacher_filter = ""
    if current_user.role == "professor":
        params["teacher_id"] = current_user.teacher_id
        teacher_filter = "AND ms.teacher_id = :teacher_id"

    rows = db.execute(
        text(
            f"""
            SELECT
                be.behavior,
                COUNT(*) AS records,
                COALESCE(SUM(be.duration_seconds), 0) AS duration_seconds
            FROM behavior_episode be
            JOIN monitoring_sessions ms ON ms.id = be.monitoring_session_id
            WHERE LOWER(TRIM(COALESCE(be.behavior, ''))) <> 'indeterminado'
            {teacher_filter}
            GROUP BY be.behavior
            ORDER BY duration_seconds DESC, records DESC, be.behavior
            """
        ),
        params,
    ).mappings().all()

    return [ReportSummaryItem(**dict(row)) for row in rows]


def list_behavior_episodes(
    db: Session,
    current_user: CurrentUser,
    student_name: str | None = None,
    behavior: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[BehaviorEpisodeRead], int]:
    params: dict[str, object] = {"limit": limit, "offset": offset}
    conditions = ["LOWER(TRIM(COALESCE(be.behavior, ''))) <> 'indeterminado'"]

    if current_user.role == "professor":
        conditions.append("ms.teacher_id = :teacher_id")
        params["teacher_id"] = current_user.teacher_id
    if student_name:
        conditions.append("be.student ILIKE :student_name")
        params["student_name"] = f"%{student_name.strip()}%"
    if behavior:
        conditions.append("be.behavior = :behavior")
        params["behavior"] = behavior

    where_clause = " AND ".join(conditions)
    total = db.execute(
        text(
            f"""
            SELECT COUNT(*)
            FROM behavior_episode be
            JOIN monitoring_sessions ms ON ms.id = be.monitoring_session_id
            WHERE {where_clause}
            """
        ),
        params,
    ).scalar_one()

    rows = db.execute(
        text(
            f"""
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
                COALESCE(be.lesson_type, ms.lesson_type) AS lesson_type,
                sub.nome AS discipline,
                tch.nome AS teacher,
                cls.nome AS class_name,
                COALESCE(cls.identificador, '') AS class_identifier
            FROM behavior_episode be
            JOIN monitoring_sessions ms ON ms.id = be.monitoring_session_id
            JOIN subjects sub ON sub.id = ms.subject_id
            JOIN classes cls ON cls.id = ms.class_id
            JOIN teachers tch ON tch.id = ms.teacher_id
            WHERE {where_clause}
            ORDER BY be.start_time DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).mappings().all()

    return [BehaviorEpisodeRead(**dict(row)) for row in rows], int(total or 0)
