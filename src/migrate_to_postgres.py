#!/usr/bin/env python3
"""
Script de Migração: SQLite → PostgreSQL
Migra dados do behavior_data.db para PostgreSQL
"""

import sqlite3
import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, UniqueConstraint
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self):
        # Configurações do SQLite (origem)
        self.sqlite_path = os.path.join(os.path.dirname(__file__), "..", "model", "behavior_data.db")
        
        # Configurações do PostgreSQL (destino)
        self.postgres_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'insightface_db'),
            'user': os.getenv('DB_USER', 'insightface_user'),
            'password': os.getenv('DB_PASSWORD', 'insightface_password')
        }
        
        self.postgres_uri = f"postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}@{self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}"
        
        # Engines
        self.sqlite_engine = None
        self.postgres_engine = None
        
    def connect_databases(self):
        """Conecta aos bancos de dados"""
        try:
            # Conectar ao SQLite
            self.sqlite_engine = create_engine(f"sqlite:///{self.sqlite_path}")
            logger.info("✅ Conectado ao SQLite")
            
            # Conectar ao PostgreSQL
            self.postgres_engine = create_engine(self.postgres_uri, pool_pre_ping=True)
            logger.info("✅ Conectado ao PostgreSQL")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            return False
    
    def create_postgres_tables(self):
        """Cria as tabelas no PostgreSQL com schema otimizado"""
        try:
            metadata = MetaData()
            
            # Tabela behavior_log (otimizada para PostgreSQL)
            behavior_log = Table(
                'behavior_log', metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('school', String(255), nullable=True),
                Column('discipline', String(255), nullable=True),
                Column('teacher', String(255), nullable=True),
                Column('student', String(255), nullable=True),
                Column('id_student', String(255), nullable=True),
                Column('behavior', String(100), nullable=True),
                Column('count', Integer, default=0),
                Column('date', String(10), nullable=True),  # YYYY-MM-DD
                Column('start_time', String(8), nullable=True),  # HH:MM:SS
                Column('end_time', String(8), nullable=True),    # HH:MM:SS
                UniqueConstraint('student', 'behavior', 'date', name='uq_behavior_log')
            )
            
            # Tabela students
            students = Table(
                'students', metadata,
                Column('id', String(255), primary_key=True),
                Column('name', String(255), nullable=True)
            )
            
            # Tabela users (já existe, mas vamos garantir)
            users = Table(
                'users', metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('cpf', String(11), unique=True, nullable=False),
                Column('nome', String(255), nullable=False),
                Column('password', String(255), nullable=False),
                Column('cidade', String(255), nullable=False),
                Column('estado', String(2), nullable=False),
                UniqueConstraint('cpf', name='uq_users_cpf')
            )
            
            # Criar todas as tabelas
            metadata.create_all(self.postgres_engine)
            logger.info("✅ Tabelas criadas no PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar tabelas: {e}")
            return False
    
    def migrate_behavior_log(self):
        """Migra dados da tabela behavior_log"""
        try:
            logger.info("🔄 Migrando behavior_log...")
            
            # Ler dados do SQLite
            query = "SELECT * FROM behavior_log"
            df = pd.read_sql_query(query, self.sqlite_engine)
            
            if df.empty:
                logger.warning("⚠️ Nenhum dado encontrado em behavior_log")
                return True
            
            # Limpar dados existentes no PostgreSQL
            with self.postgres_engine.connect() as conn:
                conn.execute(text("DELETE FROM behavior_log"))
                conn.commit()
            
            # Inserir dados no PostgreSQL
            df.to_sql('behavior_log', self.postgres_engine, if_exists='append', index=False)
            
            logger.info(f"✅ Migrados {len(df)} registros de behavior_log")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao migrar behavior_log: {e}")
            return False
    
    def migrate_students(self):
        """Migra dados da tabela students"""
        try:
            logger.info("🔄 Migrando students...")
            
            # Verificar se tabela existe no SQLite
            with self.sqlite_engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='students'"))
                if not result.fetchone():
                    logger.warning("⚠️ Tabela students não existe no SQLite")
                    return True
            
            # Ler dados do SQLite
            query = "SELECT * FROM students"
            df = pd.read_sql_query(query, self.sqlite_engine)
            
            if df.empty:
                logger.warning("⚠️ Nenhum dado encontrado em students")
                return True
            
            # Limpar dados existentes no PostgreSQL
            with self.postgres_engine.connect() as conn:
                conn.execute(text("DELETE FROM students"))
                conn.commit()
            
            # Inserir dados no PostgreSQL
            df.to_sql('students', self.postgres_engine, if_exists='append', index=False)
            
            logger.info(f"✅ Migrados {len(df)} registros de students")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao migrar students: {e}")
            return False
    
    def migrate_users(self):
        """Migra dados da tabela users"""
        try:
            logger.info("🔄 Migrando users...")
            
            # Verificar se tabela existe no SQLite
            with self.sqlite_engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'"))
                if not result.fetchone():
                    logger.warning("⚠️ Tabela users não existe no SQLite")
                    return True
            
            # Ler dados do SQLite
            query = "SELECT * FROM users"
            df = pd.read_sql_query(query, self.sqlite_engine)
            
            if df.empty:
                logger.warning("⚠️ Nenhum dado encontrado em users")
                return True
            
            # Limpar dados existentes no PostgreSQL
            with self.postgres_engine.connect() as conn:
                conn.execute(text("DELETE FROM users"))
                conn.commit()
            
            # Inserir dados no PostgreSQL
            df.to_sql('users', self.postgres_engine, if_exists='append', index=False)
            
            logger.info(f"✅ Migrados {len(df)} registros de users")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao migrar users: {e}")
            return False
    
    def verify_migration(self):
        """Verifica se a migração foi bem-sucedida"""
        try:
            logger.info("🔍 Verificando migração...")
            
            # Contar registros em cada tabela
            tables = ['behavior_log', 'students', 'users']
            
            for table in tables:
                try:
                    with self.postgres_engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        logger.info(f"📊 {table}: {count} registros")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao verificar {table}: {e}")
            
            logger.info("✅ Verificação concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na verificação: {e}")
            return False
    
    def run_migration(self):
        """Executa a migração completa"""
        logger.info("🚀 Iniciando migração SQLite → PostgreSQL")
        logger.info(f"📁 SQLite: {self.sqlite_path}")
        logger.info(f"🐘 PostgreSQL: {self.postgres_config['host']}:{self.postgres_config['port']}/{self.postgres_config['database']}")
        
        # Conectar aos bancos
        if not self.connect_databases():
            return False
        
        # Criar tabelas no PostgreSQL
        if not self.create_postgres_tables():
            return False
        
        # Migrar dados
        success = True
        success &= self.migrate_behavior_log()
        success &= self.migrate_students()
        success &= self.migrate_users()
        
        if success:
            # Verificar migração
            self.verify_migration()
            logger.info("🎉 Migração concluída com sucesso!")
        else:
            logger.error("❌ Migração falhou!")
        
        return success

def main():
    """Função principal"""
    print("🔄 Migração SQLite → PostgreSQL")
    print("=" * 50)
    
    # Verificar se arquivo SQLite existe
    sqlite_path = os.path.join(os.path.dirname(__file__), "..", "model", "behavior_data.db")
    if not os.path.exists(sqlite_path):
        print(f"❌ Arquivo SQLite não encontrado: {sqlite_path}")
        return False
    
    # Executar migração
    migrator = DatabaseMigrator()
    success = migrator.run_migration()
    
    if success:
        print("\n✅ Migração concluída!")
        print("🔧 Agora você pode atualizar o control_database.py para usar PostgreSQL")
    else:
        print("\n❌ Migração falhou!")
        print("🔍 Verifique os logs acima para mais detalhes")
    
    return success

if __name__ == "__main__":
    main()
