from pydantic import BaseModel


class ClassRead(BaseModel):
    id: int
    nome: str
    identificador: str | None = None


class StudentCreate(BaseModel):
    name: str
    matricula: str | None = None
    class_id: int | None = None


class StudentUpdate(BaseModel):
    name: str | None = None
    matricula: str | None = None
    class_id: int | None = None
    ativo: bool | None = None


class StudentRead(BaseModel):
    id: str
    name: str
    matricula: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    class_identifier: str | None = None


class StudentListResponse(BaseModel):
    items: list[StudentRead]


class ClassListResponse(BaseModel):
    items: list[ClassRead]


class NextRegistrationResponse(BaseModel):
    matricula: str
