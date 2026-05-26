from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import create_access_token, verify_password
from app.schemas.auth import LoginRequest, TokenResponse


def authenticate_user(db: Session, payload: LoginRequest) -> TokenResponse:
    row = db.execute(
        text(
            """
            SELECT id, cpf, password, role, ativo
            FROM users
            WHERE cpf = :cpf
            """
        ),
        {"cpf": payload.cpf},
    ).mappings().first()

    if row is None or not bool(row["ativo"]) or not verify_password(payload.password, row["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(subject=row["cpf"], claims={"role": row["role"] or "professor"})
    return TokenResponse(access_token=token, token_type="bearer")
