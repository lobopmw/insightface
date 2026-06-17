from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.schemas.users import CurrentUser

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    payload = decode_access_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    row = db.execute(
        text(
            """
            SELECT id, nome, cpf, cidade, estado, role, ativo
            FROM users
            WHERE cpf = :cpf
            """
        ),
        {"cpf": str(payload["sub"])},
    ).mappings().first()

    if row is None or not bool(row["ativo"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or missing user")

    teacher_row = db.execute(
        text("SELECT id FROM teachers WHERE user_id = :user_id"),
        {"user_id": row["id"]},
    ).mappings().first()

    return CurrentUser(
        id=row["id"],
        cpf=row["cpf"],
        nome=row["nome"],
        cidade=row["cidade"],
        estado=row["estado"],
        role=(row["role"] or "professor").lower(),
        ativo=bool(row["ativo"]),
        teacher_id=teacher_row["id"] if teacher_row else None,
    )
