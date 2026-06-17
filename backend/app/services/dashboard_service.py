from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.dashboard import DashboardOverview
from app.schemas.users import CurrentUser


def get_dashboard_overview(db: Session, current_user: CurrentUser) -> DashboardOverview:
    params: dict[str, object] = {}
    session_filter = ""
    student_filter = ""

    if current_user.role == "professor":
        params["teacher_id"] = current_user.teacher_id
        session_filter = "AND teacher_id = :teacher_id"
        student_filter = """
            AND class_id IN (
                SELECT class_id
                FROM teacher_subject_class
                WHERE teacher_id = :teacher_id
            )
        """

    active_sessions = db.execute(
        text(
            f"""
            SELECT COUNT(*)
            FROM monitoring_sessions
            WHERE status = 'em_andamento'
            {session_filter}
            """
        ),
        params,
    ).scalar_one()

    students_monitored = db.execute(
        text(
            f"""
            SELECT COUNT(*)
            FROM students
            WHERE ativo = TRUE
            {student_filter}
            """
        ),
        params,
    ).scalar_one()

    return DashboardOverview(
        active_sessions=int(active_sessions or 0),
        students_monitored=int(students_monitored or 0),
        alerts=0,
    )
