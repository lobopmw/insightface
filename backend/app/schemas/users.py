from pydantic import BaseModel


class CurrentUser(BaseModel):
    id: int
    cpf: str
    nome: str
    cidade: str | None = None
    estado: str | None = None
    role: str
    ativo: bool
    teacher_id: int | None = None
