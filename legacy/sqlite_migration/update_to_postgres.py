#!/usr/bin/env python3
"""
Script para atualizar arquivos Python para usar PostgreSQL
Substitui importações e configurações do SQLite para PostgreSQL
"""

import os
import shutil
import re
from pathlib import Path

def backup_original_files():
    """Cria backup dos arquivos originais"""
    print("📦 Criando backup dos arquivos originais...")
    
    backup_dir = Path("backup_sqlite")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "src/control_database.py",
        "src/main.py",
        "src/insightface_classroom.py"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)
            print(f"✅ Backup criado: {backup_path}")
    
    print("✅ Backup concluído")

def update_control_database():
    """Atualiza control_database.py para PostgreSQL"""
    print("🔄 Atualizando control_database.py...")
    
    # Substituir o arquivo original pela versão PostgreSQL
    if os.path.exists("src/control_database_postgres.py"):
        shutil.copy2("src/control_database_postgres.py", "src/control_database.py")
        print("✅ control_database.py atualizado para PostgreSQL")
    else:
        print("❌ Arquivo control_database_postgres.py não encontrado")

def update_main_py():
    """Atualiza main.py para usar PostgreSQL"""
    print("🔄 Atualizando main.py...")
    
    main_py_path = "src/main.py"
    
    if not os.path.exists(main_py_path):
        print("❌ Arquivo main.py não encontrado")
        return
    
    # Ler arquivo
    with open(main_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituir importações se necessário
    # (O main.py já usa SQLAlchemy, então não precisa de mudanças)
    
    # Adicionar suporte a variáveis de ambiente se não existir
    if "from dotenv import load_dotenv" not in content:
        # Adicionar import do dotenv no início
        lines = content.split('\n')
        import_line = "from dotenv import load_dotenv"
        load_line = "load_dotenv()"
        
        # Encontrar onde inserir
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_index = i + 1
            elif line.strip() == "":
                continue
            else:
                break
        
        # Inserir imports
        lines.insert(insert_index, import_line)
        lines.insert(insert_index + 1, load_line)
        lines.insert(insert_index + 2, "")
        
        content = '\n'.join(lines)
    
    # Salvar arquivo atualizado
    with open(main_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py atualizado")

def update_insightface_classroom():
    """Atualiza insightface_classroom.py se necessário"""
    print("🔄 Verificando insightface_classroom.py...")
    
    insightface_path = "src/insightface_classroom.py"
    
    if not os.path.exists(insightface_path):
        print("❌ Arquivo insightface_classroom.py não encontrado")
        return
    
    # Ler arquivo
    with open(insightface_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar se precisa de alterações
    if "control_database" in content:
        print("✅ insightface_classroom.py já usa control_database (compatível)")
    else:
        print("ℹ️ insightface_classroom.py não precisa de alterações")

def create_env_file():
    """Cria arquivo .env se não existir"""
    print("🔧 Configurando arquivo .env...")
    
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy2(".env.example", ".env")
            print("✅ Arquivo .env criado a partir do .env.example")
        else:
            # Criar .env básico
            env_content = """# Configurações do Banco de Dados PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=insightface_db
DB_USER=insightface_user
DB_PASSWORD=insightface_password

# Configurações da Aplicação
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ Arquivo .env criado")
    else:
        print("✅ Arquivo .env já existe")

def create_dockerfile():
    """Cria Dockerfile para a aplicação"""
    print("🐳 Criando Dockerfile...")
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    libpq-dev \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor porta
EXPOSE 8501

# Comando para iniciar
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile criado")

def show_next_steps():
    """Mostra próximos passos"""
    print("\n" + "="*60)
    print("🎉 ATUALIZAÇÃO CONCLUÍDA!")
    print("="*60)
    print()
    print("📋 Próximos passos:")
    print("1. Configure o arquivo .env com suas credenciais PostgreSQL")
    print("2. Execute a migração dos dados:")
    print("   python3 src/migrate_to_postgres.py")
    print("3. Teste a aplicação:")
    print("   streamlit run src/main.py")
    print()
    print("🐳 Para usar com Docker:")
    print("   docker-compose up -d")
    print()
    print("📁 Arquivos de backup salvos em: backup_sqlite/")
    print("🔄 Para reverter: copie os arquivos de volta de backup_sqlite/")

def main():
    """Função principal"""
    print("🔄 Atualizando projeto para PostgreSQL")
    print("="*50)
    
    try:
        backup_original_files()
        update_control_database()
        update_main_py()
        update_insightface_classroom()
        create_env_file()
        create_dockerfile()
        show_next_steps()
        
        print("\n✅ Atualização concluída com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a atualização: {e}")
        print("🔄 Verifique os logs acima para mais detalhes")

if __name__ == "__main__":
    main()
