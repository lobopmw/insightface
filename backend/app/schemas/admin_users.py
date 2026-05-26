from pydantic import BaseModel, Field


class AdminUserRead(BaseModel):
    id: int
    nome: str
    cpf: str
    cidade: str | None = None
    estado: str | None = None
    email: str | None = None
    role: str
    ativo: bool


class AdminUserCreate(BaseModel):
    cpf: str = Field(min_length=11, max_length=11)
    nome: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=6)
    cidade: str | None = None
    estado: str | None = Field(default=None, max_length=2)
    email: str | None = None
    role: str = "professor"


class AdminUserUpdate(BaseModel):
    nome: str | None = None
    cidade: str | None = None
    estado: str | None = Field(default=None, max_length=2)
    email: str | None = None
    role: str | None = None
    ativo: bool | None = None


class AdminUserListResponse(BaseModel):
    items: list[AdminUserRead]
