# Docker

Diretório reservado para a nova composição de serviços:

- backend FastAPI com suporte a CUDA/NVIDIA no estágio de IA;
- frontend React servido por Nginx;
- PostgreSQL com extensão pgvector;
- relay/streaming de câmeras quando separado do backend principal.

Arquivos iniciais:

- `backend.Dockerfile`
- `frontend.Dockerfile`
- `compose.yml`
- `docker-compose.migration.yml` preservado como rascunho anterior

Nenhum container legado foi alterado nesta etapa. O arquivo `docker-compose.yml` da raiz continua intacto.

## Nova Stack

Subir:

```bash
make compose-up
```

Ver logs:

```bash
make compose-logs
```

Criar admin:

```bash
make compose-create-admin cpf=71616845104 password='sua-senha' name='Administrador'
```

Acessos:

- Frontend/Nginx: `http://localhost:8080`
- Backend direto: `http://localhost:8000`
- Swagger: `http://localhost:8000/api/docs`
- PostgreSQL: `localhost:5432`

Guia completo: `docs/container_guide.md`.
