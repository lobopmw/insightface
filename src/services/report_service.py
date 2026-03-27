from __future__ import annotations

from datetime import timedelta
from typing import Dict

import pandas as pd

from repositories.report_repository import fetch_behavior_episodes, list_report_filter_options, list_report_students


TIME_BUCKETS = [
    (0, 6, "00:00-05:59"),
    (6, 9, "06:00-08:59"),
    (9, 12, "09:00-11:59"),
    (12, 15, "12:00-14:59"),
    (15, 18, "15:00-17:59"),
    (18, 24, "18:00-23:59"),
]

SESSION_SEGMENTS = [
    (0.0, 1 / 3, "Início da aula"),
    (1 / 3, 2 / 3, "Meio da aula"),
    (2 / 3, 1.01, "Final da aula"),
]

BEHAVIOR_LABELS = {
    "Distraido": "Distraído",
}

SOURCE_LABELS = {
    "realtime": "Tempo real",
    "simulated": "Simulado",
}


def _bucket_for_hour(hour: int) -> str:
    for start_hour, end_hour, label in TIME_BUCKETS:
        if start_hour <= hour < end_hour:
            return label
    return "Não classificado"


def _segment_within_day(relative_position: float) -> str:
    for lower, upper, label in SESSION_SEGMENTS:
        if lower <= relative_position < upper:
            return label
    return "Não classificado"


def get_available_filters(user_context: dict, filters: dict | None = None) -> dict[str, pd.DataFrame]:
    return list_report_filter_options(user_context, filters=filters)


def get_available_students(user_context: dict, filters: dict | None = None) -> pd.DataFrame:
    return list_report_students(user_context, filters=filters)


def _prepare_episodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    prepared = df.copy()
    prepared["start_time"] = pd.to_datetime(prepared["start_time"])
    prepared["end_time"] = pd.to_datetime(prepared["end_time"])
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.date
    prepared["duration_seconds"] = pd.to_numeric(prepared["duration_seconds"], errors="coerce").fillna(0.0)
    prepared["behavior"] = prepared["behavior"].replace(BEHAVIOR_LABELS)
    prepared["source"] = prepared["source"].replace(SOURCE_LABELS)
    prepared["discipline"] = prepared["discipline"].fillna("Não informada")
    prepared["teacher"] = prepared["teacher"].fillna("Não informado")
    prepared["weekday_label"] = pd.to_datetime(prepared["date"]).dt.strftime("%d/%m (%a)")
    prepared["time_bucket"] = prepared["start_time"].dt.hour.apply(_bucket_for_hour)
    prepared["period_label_daily"] = pd.to_datetime(prepared["date"]).dt.strftime("%d-%m-%Y")
    prepared["period_label_weekly"] = prepared["start_time"].dt.strftime("%G-S%V")
    prepared["period_label_monthly"] = prepared["start_time"].dt.strftime("%Y-%m")
    prepared["day_session_start"] = prepared.groupby("date")["start_time"].transform("min")
    prepared["day_session_end"] = prepared.groupby("date")["end_time"].transform("max")
    span_seconds = (
        prepared["day_session_end"] - prepared["day_session_start"]
    ).dt.total_seconds().clip(lower=1.0)
    relative_position = (
        (prepared["start_time"] - prepared["day_session_start"]).dt.total_seconds() / span_seconds
    ).fillna(0.0)
    prepared["session_segment"] = relative_position.apply(_segment_within_day)
    return prepared


def _build_behavior_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "behavior",
                "records",
                "occurrence_percentage",
                "total_duration_seconds",
                "duration_minutes",
                "duration_percentage",
            ]
        )

    summary = (
        df.groupby("behavior", as_index=False)
        .agg(
            records=("behavior", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values(["records", "total_duration_seconds", "behavior"], ascending=[False, False, True])
    )
    total_records = max(int(summary["records"].sum()), 1)
    total_duration = float(summary["total_duration_seconds"].sum())
    summary["occurrence_percentage"] = (summary["records"] / total_records * 100.0).round(2)
    summary["duration_minutes"] = (summary["total_duration_seconds"] / 60.0).round(2)
    summary["duration_percentage"] = (
        (summary["total_duration_seconds"] / total_duration * 100.0).round(2) if total_duration > 0 else 0.0
    )
    return summary.reset_index(drop=True)


def _build_daily_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "behavior", "records", "total_duration_seconds", "duration_minutes"])

    daily = (
        df.groupby(["date", "behavior"], as_index=False)
        .agg(
            records=("behavior", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values(["date", "behavior"])
    )
    daily["duration_minutes"] = (daily["total_duration_seconds"] / 60.0).round(2)
    return daily


def _build_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time_bucket", "behavior", "records", "total_duration_seconds", "duration_minutes"])

    time_dist = (
        df.groupby(["time_bucket", "behavior"], as_index=False)
        .agg(
            records=("behavior", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values(["time_bucket", "behavior"])
    )
    time_dist["duration_minutes"] = (time_dist["total_duration_seconds"] / 60.0).round(2)
    return time_dist


def _build_session_segment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["session_segment", "behavior", "records", "total_duration_seconds", "duration_minutes"]
        )

    segment_dist = (
        df.groupby(["session_segment", "behavior"], as_index=False)
        .agg(
            records=("behavior", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values(["session_segment", "behavior"])
    )
    segment_dist["duration_minutes"] = (segment_dist["total_duration_seconds"] / 60.0).round(2)
    return segment_dist


def _build_period_distribution(df: pd.DataFrame, period_mode: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period_label", "behavior", "records", "total_duration_seconds", "duration_minutes"])

    period_column_map = {
        "Diário": "period_label_daily",
        "Semanal": "period_label_weekly",
        "Mensal": "period_label_monthly",
    }
    period_column = period_column_map.get(period_mode, "period_label_monthly")
    period_df = (
        df.groupby([period_column, "behavior"], as_index=False)
        .agg(
            records=("behavior", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values([period_column, "behavior"])
    )
    period_df = period_df.rename(columns={period_column: "period_label"})
    period_df["duration_minutes"] = (period_df["total_duration_seconds"] / 60.0).round(2)
    return period_df


def _build_timeline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "date_label",
                "start_time",
                "end_time",
                "timeline_start",
                "timeline_end",
                "behavior",
                "duration_minutes",
                "discipline",
                "teacher",
            ]
        )

    timeline = df[
        ["date", "start_time", "end_time", "behavior", "duration_seconds", "discipline", "teacher", "source"]
    ].copy()
    anchor_date = pd.Timestamp("2000-01-01")
    timeline["date_label"] = pd.to_datetime(timeline["date"]).dt.strftime("%d-%m-%Y")
    timeline["timeline_start"] = anchor_date + (timeline["start_time"] - timeline["start_time"].dt.normalize())
    timeline["timeline_end"] = anchor_date + (timeline["end_time"] - timeline["end_time"].dt.normalize())
    timeline["duration_minutes"] = (timeline["duration_seconds"] / 60.0).round(2)
    return timeline.sort_values(["date", "start_time"]).reset_index(drop=True)


def _build_behavior_peak_days(daily_distribution: pd.DataFrame) -> pd.DataFrame:
    if daily_distribution.empty:
        return pd.DataFrame(columns=["behavior", "date", "records", "duration_minutes"])

    peak_days = (
        daily_distribution.sort_values(
            ["behavior", "records", "total_duration_seconds", "date"],
            ascending=[True, False, False, True],
        )
        .groupby("behavior", as_index=False)
        .head(3)
        .copy()
    )
    return peak_days[["behavior", "date", "records", "duration_minutes"]].reset_index(drop=True)


def _build_behavior_consistency(daily_distribution: pd.DataFrame, active_days: int) -> pd.DataFrame:
    if daily_distribution.empty or active_days <= 0:
        return pd.DataFrame(
            columns=[
                "behavior",
                "days_with_occurrence",
                "active_day_percentage",
                "total_records",
                "max_daily_records",
                "max_daily_share_percentage",
                "consistency_label",
            ]
        )

    consistency = (
        daily_distribution.groupby("behavior", as_index=False)
        .agg(
            days_with_occurrence=("date", "nunique"),
            total_records=("records", "sum"),
            max_daily_records=("records", "max"),
        )
        .sort_values(["total_records", "days_with_occurrence"], ascending=[False, False])
    )
    consistency["active_day_percentage"] = (
        consistency["days_with_occurrence"] / max(active_days, 1) * 100.0
    ).round(2)
    consistency["max_daily_share_percentage"] = (
        consistency["max_daily_records"] / consistency["total_records"].clip(lower=1) * 100.0
    ).round(2)

    def classify(row) -> str:
        if row["active_day_percentage"] >= 60 and row["max_daily_share_percentage"] <= 35:
            return "Regular"
        if row["active_day_percentage"] < 35 or row["max_daily_share_percentage"] >= 50:
            return "Episódica"
        return "Intermediária"

    consistency["consistency_label"] = consistency.apply(classify, axis=1)
    return consistency.reset_index(drop=True)


def _build_previous_period(start_date, end_date):
    period_length = (end_date - start_date).days + 1
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_length - 1)
    return previous_start, previous_end


def _build_comparison(current_summary: pd.DataFrame, previous_summary: pd.DataFrame) -> pd.DataFrame:
    if current_summary.empty and previous_summary.empty:
        return pd.DataFrame(columns=["behavior", "current_records", "previous_records", "records_delta"])

    comparison = current_summary[["behavior", "records", "duration_minutes"]].rename(
        columns={
            "records": "current_records",
            "duration_minutes": "current_duration_minutes",
        }
    )
    previous = previous_summary[["behavior", "records", "duration_minutes"]].rename(
        columns={
            "records": "previous_records",
            "duration_minutes": "previous_duration_minutes",
        }
    )
    comparison = comparison.merge(previous, on="behavior", how="outer")
    numeric_columns = [
        "current_records",
        "current_duration_minutes",
        "previous_records",
        "previous_duration_minutes",
    ]
    comparison[numeric_columns] = comparison[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    comparison["current_records"] = comparison["current_records"].astype(int)
    comparison["previous_records"] = comparison["previous_records"].astype(int)
    comparison["records_delta"] = comparison["current_records"] - comparison["previous_records"]
    comparison["duration_delta_minutes"] = (
        comparison["current_duration_minutes"] - comparison["previous_duration_minutes"]
    ).round(2)
    return comparison.sort_values(["current_records", "current_duration_minutes"], ascending=[False, False])


def _build_headline_metrics(df: pd.DataFrame, behavior_summary: pd.DataFrame) -> Dict[str, object]:
    if df.empty or behavior_summary.empty:
        return {
            "total_records": 0,
            "total_duration_seconds": 0.0,
            "predominant_behavior": "Sem dados",
            "active_days": 0,
            "mean_records_per_day": 0.0,
        }

    return {
        "total_records": int(len(df)),
        "total_duration_seconds": float(df["duration_seconds"].sum()),
        "predominant_behavior": behavior_summary.iloc[0]["behavior"],
        "active_days": int(df["date"].nunique()),
        "mean_records_per_day": round(float(len(df)) / max(int(df["date"].nunique()), 1), 2),
    }


def generate_report_data(user_context: dict, filters: dict, period_mode: str) -> Dict[str, object]:
    current_filters = dict(filters)
    current_df = _prepare_episodes(fetch_behavior_episodes(user_context, current_filters))

    previous_start, previous_end = _build_previous_period(filters["start_date"], filters["end_date"])
    previous_filters = dict(filters)
    previous_filters["start_date"] = previous_start
    previous_filters["end_date"] = previous_end
    previous_df = _prepare_episodes(fetch_behavior_episodes(user_context, previous_filters))

    current_summary = _build_behavior_summary(current_df)
    previous_summary = _build_behavior_summary(previous_df)
    daily_distribution = _build_daily_distribution(current_df)
    time_distribution = _build_time_distribution(current_df)
    session_segment_distribution = _build_session_segment_distribution(current_df)
    period_distribution = _build_period_distribution(current_df, period_mode=period_mode)
    timeline = _build_timeline(current_df)
    peak_days = _build_behavior_peak_days(daily_distribution)
    comparison = _build_comparison(current_summary, previous_summary)
    headline_metrics = _build_headline_metrics(current_df, current_summary)
    behavior_consistency = _build_behavior_consistency(daily_distribution, headline_metrics["active_days"])

    return {
        "student": filters.get("student_name"),
        "period_mode": period_mode,
        "start_date": filters["start_date"],
        "end_date": filters["end_date"],
        "previous_start_date": previous_start,
        "previous_end_date": previous_end,
        "episodes": current_df,
        "behavior_summary": current_summary,
        "daily_distribution": daily_distribution,
        "time_distribution": time_distribution,
        "session_segment_distribution": session_segment_distribution,
        "period_distribution": period_distribution,
        "timeline": timeline,
        "peak_days": peak_days,
        "comparison": comparison,
        "headline_metrics": headline_metrics,
        "behavior_consistency": behavior_consistency,
    }
