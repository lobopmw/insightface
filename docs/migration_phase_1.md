# Migração Fase 1

Objetivo desta fase: criar a base frontend/backend separada sem quebrar o Streamlit.

## Criado

- `backend/`: aplicação FastAPI mínima com `/api/health`, rotas stub, configuração, segurança JWT e WebSocket.
- `frontend/`: aplicação React + Vite + TypeScript com Tailwind, React Query, Zustand, Recharts e cliente WebSocket.
- `docker/`, `nginx/` e `docs/`: diretórios preparados para a próxima fase operacional.
- `Makefile`: comandos auxiliares para a nova stack.
- `backend/migrations/`: estrutura Alembic preparada para futuras alterações de schema.

## APIs preparadas

- `GET /api/health`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/dashboard/overview`
- `GET /api/students`
- `POST /api/students`
- `GET /api/students/classes`
- `GET /api/students/next-registration`
- `GET /api/students/{student_id}`
- `PATCH /api/students/{student_id}`
- `DELETE /api/students/{student_id}`
- `GET /api/reports/summary`
- `GET /api/reports/episodes`
- `GET /api/monitoring/status`
- `GET /api/monitoring/options`
- `POST /api/monitoring/sessions`
- `GET /api/monitoring/sessions/{session_id}`
- `POST /api/monitoring/sessions/{session_id}/close`
- `WS /api/ws/monitoring`
- `GET /api/users`
- `POST /api/users`
- `PATCH /api/users/{user_id}`

## Progresso incremental posterior

- A tela de login já autentica contra `POST /api/auth/login`.
- O dashboard já consome `GET /api/dashboard/overview`.
- A página de alunos já consome `GET /api/students`.
- A página de relatórios já consome `GET /api/reports/summary`.
- A página de monitoramento já carrega opções e cria/encerra sessões reais.
- O backend passou a ter camada `services/` para reduzir SQL dentro das rotas.
- O cadastro administrativo de alunos já permite criar, listar e desativar alunos na nova UI.
- A lógica de gerenciamento de episódios comportamentais foi copiada para `backend/app/ai/behavior/episode_manager.py`.
- A persistência de episódios foi preparada em `backend/app/services/behavior_service.py`.
- A base administrativa de usuários foi migrada para `/api/users` e aparece em Configurações para administradores.

## Remoção do legado

Nenhum arquivo Streamlit foi removido nesta fase porque a nova stack ainda não cobre IA, captura, relatórios completos e operação em tempo real. A política de remoção está em `docs/legacy_deprecation_plan.md`.

## Banco

Nenhuma tabela foi alterada nesta fase. A estrutura Alembic foi criada para que próximas mudanças sejam feitas por migrations versionadas.

## Não alterado

- Algoritmos YOLO, InsightFace/ArcFace e PGVector.
- Schema do banco de dados.
- Dockerfile e `docker-compose.yml` legados.
- Aplicação Streamlit em `src/`.

## Execução local prevista

Backend:

```bash
make backend-install-dev
make backend-test
make backend-dev
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Ou pela raiz:

```bash
make backend-dev
make frontend-install
make frontend-build
make frontend-dev
```
