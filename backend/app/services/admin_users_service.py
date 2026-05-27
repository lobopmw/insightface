from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.schemas.admin_users import AdminUserCreate, AdminUserRead, AdminUserUpdate
from app.schemas.users import CurrentUser


def ensure_admin(current_user: CurrentUser) -> None:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")


def sync_teacher_profile(db: Session, user_id: int, nome: str, role: str | None) -> None:
    if (role or "professor").lower() != "professor":
        return
    db.execute(
        text(
            """
            INSERT INTO teachers (user_id, nome)
            VALUES (:user_id, :nome)
            ON CONFLICT (user_id)
            DO UPDATE SET nome = EXCLUDED.nome
            """
        ),
        {"user_id": user_id, "nome": nome},
    )


def list_users(db: Session, current_user: CurrentUser) -> list[AdminUserRead]:
    ensure_admin(current_user)
    rows = db.execute(
        text(
            """
            SELECT id, nome, cpf, cidade, estado, email, role, ativo
            FROM users
            ORDER BY nome
            """
        )
    ).mappings().all()
    return [AdminUserRead(**dict(row)) for row in rows]


def create_user(db: Session, current_user: CurrentUser, payload: AdminUserCreate) -> AdminUserRead:
    ensure_admin(current_user)
    existing = db.execute(text("SELECT 1 FROM users WHERE cpf = :cpf"), {"cpf": payload.cpf}).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="CPF already exists")

    row = db.execute(
        text(
            """
            INSERT INTO users (cpf, nome, password, cidade, estado, email, role, ativo)
            VALUES (:cpf, :nome, :password, :cidade, :estado, :email, :role, TRUE)
            RETURNING id, nome, cpf, cidade, estado, email, role, ativo
            """
        ),
        {
            "cpf": payload.cpf,
            "nome": payload.nome.strip(),
            "password": hash_password(payload.password),
            "cidade": payload.cidade,
            "estado": payload.estado,
            "email": str(payload.email) if payload.email else None,
            "role": payload.role,
        },
    ).mappings().one()

    sync_teacher_profile(db, int(row["id"]), row["nome"], row["role"])

    db.commit()
    return AdminUserRead(**dict(row))


def update_user(db: Session, current_user: CurrentUser, user_id: int, payload: AdminUserUpdate) -> AdminUserRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            UPDATE users
            SET
                nome = COALESCE(:nome, nome),
                cidade = COALESCE(:cidade, cidade),
                estado = COALESCE(:estado, estado),
                email = COALESCE(:email, email),
                role = COALESCE(:role, role),
                ativo = COALESCE(:ativo, ativo)
            WHERE id = :user_id
            RETURNING id, nome, cpf, cidade, estado, email, role, ativo
            """
        ),
        {
            "user_id": user_id,
            "nome": payload.nome.strip() if payload.nome else None,
            "cidade": payload.cidade,
            "estado": payload.estado,
            "email": str(payload.email) if payload.email else None,
            "role": payload.role,
            "ativo": payload.ativo,
        },
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    sync_teacher_profile(db, int(row["id"]), row["nome"], row["role"])
    db.commit()
    return AdminUserRead(**dict(row))
