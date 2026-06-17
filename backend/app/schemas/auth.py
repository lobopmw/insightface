from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    cpf: str = Field(min_length=11, max_length=11)
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
