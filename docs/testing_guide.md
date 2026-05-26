# Guia de Testes da Nova Stack

Este guia cobre a stack React/FastAPI criada em paralelo ao Streamlit.

## Pré-requisitos

- Python 3.12 ou superior.
- Node.js 20 ou superior com `npm`.
- PostgreSQL com extensão `pgvector`.
- Variáveis do `.env` da raiz apontando para o banco atual.

Neste ambiente, o Node foi instalado localmente em `.local/node/node-v22.22.3-linux-x64`, e o `Makefile` já usa esse binário automaticamente.

## Backend

Instalar dependências:

```bash
make backend-install-dev
```

Rodar checagem sintática:

```bash
make backend-check
```

Rodar testes:

```bash
make backend-test
```

Subir API:

```bash
make backend-dev
```

Validar health:

```bash
curl http://localhost:8000/api/health
```

Resposta esperada:

```json
{"status":"ok","service":"InsightFace Classroom API"}
```

Documentação interativa:

```text
http://localhost:8000/api/docs
```

## Frontend

Instalar dependências:

```bash
make frontend-install
```

Build/typecheck:

```bash
make frontend-build
```

Subir Vite:

```bash
make frontend-dev
```

Abrir:

```text
http://localhost:5173
```

Build validado neste ambiente:

```bash
make frontend-build
```

## Fluxo Manual Recomendado

1. Acessar `/api/health`.
2. Fazer login na interface React com um usuário existente do banco.
3. Conferir Dashboard.
4. Abrir Alunos, cadastrar um aluno e desativá-lo.
5. Abrir Monitoramento, selecionar disciplina/turma e criar uma sessão.
6. Encerrar a sessão.
7. Abrir Relatórios e conferir resumo/listagem de episódios.
8. Entrar como admin e verificar usuários em Configurações.

## Criar Admin de Teste

Com PostgreSQL rodando e `.env` apontando para o banco:

```bash
make backend-create-admin cpf=71616845104 password='sua-senha' name='Administrador'
```

O comando atualiza o usuário existente para `admin` ou cria um novo registro em `users`.

## Observações

- A aplicação Streamlit antiga continua disponível e não foi removida.
- A IA em tempo real ainda não foi conectada ao backend FastAPI.
- Alterações de schema devem ser feitas via Alembic em `backend/migrations`.
- Neste ambiente, a API foi validada em `http://127.0.0.1:8000/api/health`.
- Para testar via containers, use `docs/container_guide.md`.
