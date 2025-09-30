#!/bin/bash

# Script de configuração para migração SQLite → PostgreSQL
# Este script automatiza todo o processo de migração

set -e  # Parar em caso de erro

echo "🚀 Configurando migração SQLite → PostgreSQL"
echo "=============================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens coloridas
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar se Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker não está instalado. Por favor, instale o Docker primeiro."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose não está instalado. Por favor, instale o Docker Compose primeiro."
        exit 1
    fi
    
    print_success "Docker e Docker Compose estão instalados"
}

# Verificar se Python está instalado
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 não está instalado."
        exit 1
    fi
    
    print_success "Python 3 está instalado"
}

# Instalar dependências Python
install_dependencies() {
    print_status "Instalando dependências Python..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependências Python instaladas"
    else
        print_error "Arquivo requirements.txt não encontrado"
        exit 1
    fi
}

# Criar arquivo .env se não existir
create_env_file() {
    if [ ! -f ".env" ]; then
        print_status "Criando arquivo .env..."
        cp .env.example .env
        print_success "Arquivo .env criado"
        print_warning "Edite o arquivo .env com suas configurações antes de continuar"
    else
        print_success "Arquivo .env já existe"
    fi
}

# Subir PostgreSQL
start_postgres() {
    print_status "Iniciando PostgreSQL container..."
    
    # Parar containers existentes
    docker-compose down 2>/dev/null || true
    
    # Subir apenas o PostgreSQL
    docker-compose up -d postgres
    
    # Aguardar PostgreSQL ficar pronto
    print_status "Aguardando PostgreSQL ficar pronto..."
    sleep 10
    
    # Verificar se PostgreSQL está rodando
    if docker-compose ps postgres | grep -q "Up"; then
        print_success "PostgreSQL está rodando"
    else
        print_error "Falha ao iniciar PostgreSQL"
        exit 1
    fi
}

# Executar migração
run_migration() {
    print_status "Executando migração dos dados..."
    
    if [ -f "src/migrate_to_postgres.py" ]; then
        cd src
        python3 migrate_to_postgres.py
        cd ..
        print_success "Migração concluída"
    else
        print_error "Script de migração não encontrado"
        exit 1
    fi
}

# Verificar migração
verify_migration() {
    print_status "Verificando migração..."
    
    # Conectar ao PostgreSQL e verificar tabelas
    docker exec insightface_postgres psql -U insightface_user -d insightface_db -c "\dt"
    
    print_success "Verificação concluída"
}

# Mostrar instruções finais
show_final_instructions() {
    echo ""
    echo "🎉 Configuração concluída!"
    echo "=========================="
    echo ""
    echo "📋 Próximos passos:"
    echo "1. Edite o arquivo .env com suas configurações"
    echo "2. Atualize seus arquivos Python para usar PostgreSQL:"
    echo "   - Substitua 'control_database.py' por 'control_database_postgres.py'"
    echo "   - Atualize as importações em 'main.py' e 'insightface_classroom.py'"
    echo ""
    echo "🔧 Comandos úteis:"
    echo "   - Parar PostgreSQL: docker-compose down"
    echo "   - Ver logs: docker-compose logs postgres"
    echo "   - Conectar ao banco: docker exec -it insightface_postgres psql -U insightface_user -d insightface_db"
    echo ""
    echo "📊 Status do PostgreSQL:"
    docker-compose ps postgres
}

# Função principal
main() {
    echo "Iniciando configuração..."
    
    check_docker
    check_python
    install_dependencies
    create_env_file
    
    echo ""
    print_warning "IMPORTANTE: Edite o arquivo .env antes de continuar!"
    read -p "Pressione Enter quando terminar de editar o .env..."
    
    start_postgres
    run_migration
    verify_migration
    show_final_instructions
}

# Executar script
main "$@"
