-- Script de inicialização do PostgreSQL para InsightFace
-- Este script é executado automaticamente quando o container é criado

-- Configurações otimizadas para o projeto
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Criar extensões úteis
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Configurar timezone
SET timezone = 'America/Sao_Paulo';

-- Comentários sobre o banco
COMMENT ON DATABASE insightface_db IS 'Banco de dados para sistema de monitoramento de comportamentos em sala de aula';
