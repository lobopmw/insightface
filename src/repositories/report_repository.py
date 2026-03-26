from __future__ import annotations

import pandas as pd

from control_database_postgres import engine


def list_report_students() -> pd.DataFrame:
    query = """
        SELECT
            student,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(*) AS total_episodes
        FROM behavior_episode
        WHERE student IS NOT NULL
          AND TRIM(student) <> ''
        GROUP BY student
        ORDER BY student
    """
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def fetch_behavior_episodes(student: str, start_date, end_date) -> pd.DataFrame:
    query = """
        SELECT
            school,
            discipline,
            teacher,
            student,
            id_student,
            behavior,
            start_time,
            end_time,
            duration_seconds,
            date,
            source
        FROM behavior_episode
        WHERE student = %(student)s
          AND date >= %(start_date)s
          AND date <= %(end_date)s
        ORDER BY start_time ASC
    """
    with engine.connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={
                "student": student,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
