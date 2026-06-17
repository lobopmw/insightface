# Guia de Containers da Nova Stack

Este guia sobe apenas a nova arquitetura React + FastAPI + PostgreSQL/pgvector. A stack Streamlit antiga da raiz permanece preservada.

## Serviços

- `postgres`: PostgreSQL com pgvector.
- `backend`: FastAPI, aplica migrations Alembic e sobe em `8000`.
- `frontend`: Nginx servindo React e proxy para `/api` e WebSocket.
- `admin-init`: ferramenta opcional para criar/atualizar usuário admin.

## Permissão Docker

Se `docker compose` retornar erro de permissão em `/var/run/docker.sock`, use uma destas opções:

```bash
sudo docker compose version
```

Ou corrija permanentemente o grupo do usuário:

```bash
sudo usermod -aG docker $USER
```

Depois faça logout/login no Ubuntu para o grupo valer.

## Subir

```bash
make compose-up
```

Ver status:

```bash
make compose-ps
```

Ver logs:

```bash
make compose-logs
```

Parar:

```bash
make compose-down
```

## Acessos

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`
- Swagger: `http://localhost:8000/api/docs`
- Health: `http://localhost:8000/api/health`
- PostgreSQL: `localhost:5432`

## Criar Admin

Com os containers ativos:

```bash
make compose-create-admin cpf=71616845104 password='sua-senha' name='Administrador'
```

O comando cria o usuário se não existir ou atualiza o CPF existente para `role='admin'` e `ativo=TRUE`.

## Variáveis

O Compose usa `docker/.env.example` por padrão. Para personalizar sem alterar o exemplo:

```bash
cp docker/.env.example docker/.env
```

Depois rode explicitamente:

```bash
COMPOSE="docker compose -p insightface-modern --env-file docker/.env -f docker/compose.yml" make compose-up
```
