from __future__ import annotations

import os
import sys

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.core.security import hash_password
from app.db.session import SessionLocal


def main() -> int:
    cpf = os.getenv("ADMIN_CPF", "").strip()
    password = os.getenv("ADMIN_PASSWORD", "")
    name = os.getenv("ADMIN_NAME", "Administrador").strip() or "Administrador"

    if len(cpf) != 11 or not cpf.isdigit():
        print("ADMIN_CPF deve conter 11 dígitos.", file=sys.stderr)
        return 2
    if not password:
        print("ADMIN_PASSWORD é obrigatório.", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        password_hash = hash_password(password)
        existing = db.execute(
            text("SELECT id FROM users WHERE cpf = :cpf"),
            {"cpf": cpf},
        ).mappings().first()

        if existing:
            db.execute(
                text(
                    """
                    UPDATE users
                    SET nome = COALESCE(NULLIF(:name, ''), nome),
                        password = :password,
                        role = 'admin',
                        ativo = TRUE
                    WHERE cpf = :cpf
                    """
                ),
                {"cpf": cpf, "password": password_hash, "name": name},
            )
            action = "updated"
        else:
            db.execute(
                text(
                    """
                    INSERT INTO users (cpf, nome, password, cidade, estado, email, role, ativo)
                    VALUES (:cpf, :name, :password, NULL, NULL, NULL, 'admin', TRUE)
                    """
                ),
                {"cpf": cpf, "password": password_hash, "name": name},
            )
            action = "created"

        db.commit()
        print(f"admin_user_{action}")
        return 0
    except OperationalError as exc:
        db.rollback()
        print(f"database_unavailable: {exc.orig}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
