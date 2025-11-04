# 🔄 Guia de Migração SQLite → PostgreSQL

Este guia mostra como migrar seu projeto InsightFace do SQLite para PostgreSQL.

## 📋 **Resumo das Alterações**

### **Arquivos Criados/Modificados:**

1. **Scripts de Migração:**
   - `src/migrate_to_postgres.py` - Script principal de migração
   - `update_to_postgres.py` - Atualiza arquivos Python
   - `setup_postgres.sh` - Script de configuração automatizada

2. **Configuração PostgreSQL:**
   - `control_database_postgres.py` - Versão PostgreSQL do control_database.py
   - `docker-compose.yml` - Configuração do container PostgreSQL
   - `init.sql` - Script de inicialização do PostgreSQL
   - `.env.example` - Exemplo de variáveis de ambiente

3. **Dependências:**
   - `requirements.txt` - Atualizado com psycopg2-binary e python-dotenv

## 🚀 **Processo de Migração**

### **Opção 1: Migração Automatizada (Recomendada)**

```bash
# 1. Tornar o script executável
chmod +x setup_postgres.sh

# 2. Executar configuração automatizada
./setup_postgres.sh
```

### **Opção 2: Migração Manual**

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar PostgreSQL
docker-compose up -d postgres

# 3. Executar migração
python3 src/migrate_to_postgres.py

# 4. Atualizar arquivos Python
python3 update_to_postgres.py
```

## 🔧 **Principais Alterações no Código**

### **1. Configuração do Banco (`control_database.py`)**

**Antes (SQLite):**
```python
import sqlite3
DB_PATH = os.path.join(DB_DIR, "behavior_data.db")
DATABASE_URI = f"sqlite:///{DB_PATH}"
conn = sqlite3.connect(DB_PATH)
```

**Depois (PostgreSQL):**
```python
import psycopg2
from psycopg2.extras import RealDictCursor

# Configurações via variáveis de ambiente
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "insightface_db")
DB_USER = os.getenv("DB_USER", "insightface_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "insightface_password")

DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Conexão PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, database=DB_NAME,
    user=DB_USER, password=DB_PASSWORD
)
```

### **2. Schema das Tabelas**

**SQLite:**
```sql
CREATE TABLE behavior_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school TEXT,
    behavior TEXT,
    -- ...
    UNIQUE(student, behavior, date)
)
```

**PostgreSQL:**
```sql
CREATE TABLE behavior_log (
    id SERIAL PRIMARY KEY,
    school VARCHAR(255),
    behavior VARCHAR(100),
    -- ...
    CONSTRAINT uq_behavior_log UNIQUE(student, behavior, date)
)
```

### **3. Queries SQL**

**SQLite:**
```python
cursor.execute('''
    INSERT INTO behavior_log (...) VALUES (?, ?, ?)
    ON CONFLICT(student, behavior, date) DO UPDATE SET count = count + 1
''', (value1, value2, value3))
```

**PostgreSQL:**
```python
cursor.execute('''
    INSERT INTO behavior_log (...) VALUES (%s, %s, %s)
    ON CONFLICT (student, behavior, date) 
    DO UPDATE SET count = behavior_log.count + 1
''', (value1, value2, value3))
```

## 🐳 **Configuração com Docker**

### **docker-compose.yml:**
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: insightface_db
      POSTGRES_USER: insightface_user
      POSTGRES_PASSWORD: insightface_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

### **Comandos Docker:**
```bash
# Subir PostgreSQL
docker-compose up -d postgres

# Ver logs
docker-compose logs postgres

# Conectar ao banco
docker exec -it insightface_postgres psql -U insightface_user -d insightface_db

# Parar
docker-compose down
```

## 📊 **Verificação da Migração**

### **1. Verificar Dados Migrados:**
```sql
-- Conectar ao PostgreSQL
docker exec -it insightface_postgres psql -U insightface_user -d insightface_db

-- Verificar tabelas
\dt

-- Contar registros
SELECT COUNT(*) FROM behavior_log;
SELECT COUNT(*) FROM students;
SELECT COUNT(*) FROM users;
```

### **2. Testar Aplicação:**
```bash
# Configurar variáveis de ambiente
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=insightface_db
export DB_USER=insightface_user
export DB_PASSWORD=insightface_password

# Executar aplicação
streamlit run src/main.py
```

## 🔄 **Rollback (Reversão)**

Se precisar voltar ao SQLite:

```bash
# 1. Parar PostgreSQL
docker-compose down

# 2. Restaurar arquivos originais
cp backup_sqlite/control_database.py src/
cp backup_sqlite/main.py src/
cp backup_sqlite/insightface_classroom.py src/

# 3. Remover dependências PostgreSQL
pip uninstall psycopg2-binary python-dotenv
```

## 🎯 **Vantagens da Migração**

1. **Performance:** PostgreSQL é mais rápido para consultas complexas
2. **Escalabilidade:** Suporte a múltiplos usuários simultâneos
3. **Recursos Avançados:** JSON, arrays, funções personalizadas
4. **Backup/Restore:** Ferramentas robustas de backup
5. **Monitoramento:** Estatísticas detalhadas de performance
6. **Concorrência:** Melhor controle de transações

## 🚨 **Troubleshooting**

### **Problemas Comuns:**

1. **Erro de Conexão:**
   ```bash
   # Verificar se PostgreSQL está rodando
   docker-compose ps postgres
   
   # Ver logs
   docker-compose logs postgres
   ```

2. **Erro de Permissão:**
   ```bash
   # Verificar variáveis de ambiente
   cat .env
   ```

3. **Dados Não Migrados:**
   ```bash
   # Executar migração novamente
   python3 src/migrate_to_postgres.py
   ```

4. **Erro de Import:**
   ```bash
   # Instalar dependências
   pip install -r requirements.txt
   ```

## 📞 **Suporte**

Se encontrar problemas:

1. Verifique os logs: `docker-compose logs postgres`
2. Confirme as variáveis de ambiente no arquivo `.env`
3. Execute o script de migração novamente
4. Verifique se todos os dados foram migrados corretamente

---

**✅ Migração concluída com sucesso!**
