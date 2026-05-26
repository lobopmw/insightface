from datetime import date, datetime

from pydantic import BaseModel


class ReportSummaryItem(BaseModel):
    behavior: str
    records: int
    duration_seconds: float


class ReportSummaryResponse(BaseModel):
    items: list[ReportSummaryItem]


class BehaviorEpisodeRead(BaseModel):
    id: int
    monitoring_session_id: int | None = None
    student_id: str | None = None
    student: str
    behavior: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    date: date
    source: str | None = None
    lesson_type: str | None = None
    discipline: str | None = None
    teacher: str | None = None
    class_name: str | None = None
    class_identifier: str | None = None


class BehaviorEpisodeListResponse(BaseModel):
    items: list[BehaviorEpisodeRead]
    total: int
