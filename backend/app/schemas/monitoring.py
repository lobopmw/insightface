from datetime import datetime

from pydantic import BaseModel, Field


class MonitoringStatusResponse(BaseModel):
    status: str


class MonitoringSubject(BaseModel):
    id: int
    nome: str


class MonitoringClass(BaseModel):
    id: int
    nome: str
    identificador: str | None = None


class MonitoringOptionsResponse(BaseModel):
    subjects: list[MonitoringSubject]
    classes: list[MonitoringClass]
    lesson_types: list[str]


class MonitoringSessionCreate(BaseModel):
    subject_id: int
    class_id: int
    lesson_type: str = Field(default="Exposição", min_length=1, max_length=80)


class MonitoringSessionRead(BaseModel):
    id: int
    status: str


class MonitoringSessionClose(BaseModel):
    status: str = "encerrada"


class MonitoringSessionSummary(BaseModel):
    id: int
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    lesson_type: str | None = None
