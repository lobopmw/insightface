from __future__ import annotations

import pandas as pd

from control_database_postgres import fetch_behavior_dataframe, list_behavior_filter_options


def list_report_filter_options(user_context: dict, filters: dict | None = None) -> dict[str, pd.DataFrame]:
    return list_behavior_filter_options(user_context, filters=filters)


def list_report_students(user_context: dict, filters: dict | None = None) -> pd.DataFrame:
    return list_report_filter_options(user_context, filters=filters)["students"]


def fetch_behavior_episodes(user_context: dict, filters: dict) -> pd.DataFrame:
    return fetch_behavior_dataframe(user_context, filters=filters)
