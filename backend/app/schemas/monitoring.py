from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class MonitoringStatusResponse(BaseModel):
    status: str
    session_id: int | None = None
    camera_status: str = "idle"
    websocket_clients: int = 0
    last_event: str | None = None
    error: str | None = None
    student_name: str = "Aguardando identificação"
    behavior: str = "Aguardando"
    confidence: float = 0
    timestamp: datetime | None = None


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


class MonitoringStartRequest(BaseModel):
    camera_id: int | None = None
    disciplina_id: int | None = None
    turma_id: int | None = None
    tipo_aula: str | None = None
    subject_id: int | None = None
    class_id: int | None = None
    lesson_type: str | None = None

    @model_validator(mode="after")
    def normalize_legacy_and_new_names(self) -> "MonitoringStartRequest":
        self.subject_id = self.subject_id or self.disciplina_id
        self.class_id = self.class_id or self.turma_id
        self.lesson_type = self.lesson_type or self.tipo_aula or "Exposição"
        return self

    def to_session_create(self) -> MonitoringSessionCreate:
        if self.subject_id is None or self.class_id is None:
            raise ValueError("subject_id/disciplina_id and class_id/turma_id are required")
        return MonitoringSessionCreate(
            subject_id=self.subject_id,
            class_id=self.class_id,
            lesson_type=self.lesson_type or "Exposição",
        )


class MonitoringStopRequest(BaseModel):
    status: str = "encerrada"
