#!/usr/bin/env python3
"""Utility CLI to ensure PostgreSQL schema is created before starting the app."""

import sys
from psycopg2 import OperationalError

from control_database_postgres import user_table, connect_database


def ensure_schema():
    """Create the expected tables if they are missing."""
    user_table()
    with connect_database():
        pass


def main():
    print("🔄 Verificando conexão com PostgreSQL...")
    try:
        ensure_schema()
    except OperationalError as exc:
        print("❌ Não foi possível conectar ao banco. Confira as variáveis de ambiente e se o serviço está ativo.")
        raise SystemExit(exc) from exc
    print("✅ Esquema verificado/criado com sucesso.")


if __name__ == "__main__":
    sys.exit(main())
