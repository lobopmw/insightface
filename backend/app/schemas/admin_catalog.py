from pydantic import BaseModel, Field


class SubjectRead(BaseModel):
    id: int
    nome: str


class SubjectCreate(BaseModel):
    nome: str = Field(min_length=1, max_length=255)


class SubjectUpdate(BaseModel):
    nome: str | None = Field(default=None, max_length=255)


class ClassRead(BaseModel):
    id: int
    nome: str
    identificador: str | None = None


class ClassCreate(BaseModel):
    nome: str = Field(min_length=1, max_length=255)
    identificador: str | None = Field(default=None, max_length=255)


class ClassUpdate(BaseModel):
    nome: str | None = Field(default=None, max_length=255)
    identificador: str | None = Field(default=None, max_length=255)


class TeacherRead(BaseModel):
    id: int
    user_id: int
    nome: str
    cpf: str
    email: str | None = None
    ativo: bool


class AssignmentRead(BaseModel):
    id: int
    teacher_id: int
    teacher_name: str
    subject_id: int
    subject_name: str
    class_id: int
    class_name: str
    class_identifier: str | None = None


class AssignmentCreate(BaseModel):
    teacher_id: int
    subject_id: int
    class_id: int


class CatalogResponse(BaseModel):
    subjects: list[SubjectRead]
    classes: list[ClassRead]
    teachers: list[TeacherRead]
    assignments: list[AssignmentRead]
