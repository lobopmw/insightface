import csv
import hashlib
import os

# Caminho absoluto para o arquivo de mapeamento
MAPPING_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mapeamento_alunos.csv'))

def gerar_hash_nome_matricula(nome, matricula):
    """
    Gera um hash SHA-256 baseado no nome e matrícula do aluno.
    """
    base = f"{nome}_{matricula}".lower().strip()
    return hashlib.sha256(base.encode('utf-8')).hexdigest()

def salvar_mapeamento(nome, matricula):
    """
    Salva o mapeamento entre nome/matrícula e hash no arquivo CSV.
    Se já existir, retorna o hash existente.
    """
    hash_pasta = gerar_hash_nome_matricula(nome, matricula)

    # Cria o CSV se não existir
    if not os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['nome', 'matricula', 'hash'])

    # Verifica se já existe o aluno no mapeamento
    with open(MAPPING_FILE, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['nome'] == nome and row['matricula'] == matricula:
                return row['hash']  # Já existente

    # Salva novo mapeamento
    with open(MAPPING_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([nome, matricula, hash_pasta])

    return hash_pasta
