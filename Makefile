PYTHON ?= .venv/bin/python
NODE_HOME ?= $(CURDIR)/.local/node/node-v22.22.3-linux-x64
NPM ?= PATH="$(NODE_HOME)/bin:$$PATH" npm
NODE ?= PATH="$(NODE_HOME)/bin:$$PATH" node

COMPOSE ?= docker compose -p insightface-modern --env-file docker/.env.example -f docker/compose.yml
SUDO_COMPOSE ?= sudo docker compose -p insightface-modern --env-file docker/.env.example -f docker/compose.yml

.PHONY: backend-install backend-install-dev backend-dev frontend-install frontend-dev frontend-build frontend-node-version backend-check backend-test compose-up compose-build compose-down compose-logs compose-ps compose-create-admin sudo-compose-up sudo-compose-down sudo-compose-logs sudo-compose-ps sudo-compose-create-admin migration-compose-up migration-compose-down

backend-install:
	$(PYTHON) -m pip install -r backend/requirements.txt

backend-install-dev:
	$(PYTHON) -m pip install -r backend/requirements-dev.txt

backend-dev:
	$(PYTHON) -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

frontend-install:
	cd frontend && $(NPM) install

frontend-dev:
	cd frontend && $(NPM) run dev

frontend-build:
	cd frontend && $(NPM) run build

frontend-node-version:
	$(NODE) --version
	$(NPM) --version

backend-check:
	$(PYTHON) -m py_compile \
		backend/app/main.py \
		backend/app/api/deps.py \
		backend/app/api/routes_auth.py \
		backend/app/api/routes_students.py \
		backend/app/api/routes_monitoring.py \
		backend/app/api/routes_reports.py \
		backend/app/api/routes_dashboard.py \
		backend/app/api/routes_users.py \
		backend/app/core/config.py \
		backend/app/core/security.py \
		backend/app/db/session.py \
		backend/app/services/students_service.py \
		backend/app/services/auth_service.py \
		backend/app/services/dashboard_service.py \
		backend/app/services/reports_service.py \
		backend/app/services/monitoring_service.py \
		backend/app/services/behavior_service.py \
		backend/app/services/admin_users_service.py \
		backend/app/ai/behavior/episode_manager.py \
		backend/app/schemas/students.py \
		backend/app/schemas/dashboard.py \
		backend/app/schemas/reports.py \
		backend/app/schemas/monitoring.py \
		backend/app/schemas/users.py \
		backend/app/schemas/admin_users.py

backend-test:
	$(PYTHON) -m pytest -q backend/tests

backend-create-admin:
	@PYTHONPATH=backend ADMIN_CPF="$(cpf)" ADMIN_PASSWORD="$(password)" ADMIN_NAME="$(name)" $(PYTHON) backend/scripts/create_admin_user.py

compose-build:
	$(COMPOSE) build

compose-up:
	$(COMPOSE) up -d --build

compose-down:
	$(COMPOSE) down

compose-logs:
	$(COMPOSE) logs -f

compose-ps:
	$(COMPOSE) ps

compose-create-admin:
	@$(COMPOSE) --profile tools run --rm \
		-e ADMIN_CPF="$(cpf)" \
		-e ADMIN_PASSWORD="$(password)" \
		-e ADMIN_NAME="$(name)" \
		admin-init

sudo-compose-up:
	$(SUDO_COMPOSE) up -d --build

sudo-compose-down:
	$(SUDO_COMPOSE) down

sudo-compose-logs:
	$(SUDO_COMPOSE) logs -f

sudo-compose-ps:
	$(SUDO_COMPOSE) ps

sudo-compose-create-admin:
	@$(SUDO_COMPOSE) --profile tools run --rm \
		-e ADMIN_CPF="$(cpf)" \
		-e ADMIN_PASSWORD="$(password)" \
		-e ADMIN_NAME="$(name)" \
		admin-init

migration-compose-up:
	docker compose -f docker/docker-compose.migration.yml up --build

migration-compose-down:
	docker compose -f docker/docker-compose.migration.yml down

backend-migration-new:
	cd backend && ../$(PYTHON) -m alembic revision -m "$(name)"

backend-migrate:
	cd backend && ../$(PYTHON) -m alembic upgrade head
