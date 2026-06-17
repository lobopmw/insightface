# Nginx

Diretório reservado para o proxy reverso futuro.

Rotas previstas:

- `/` para o frontend React estático;
- `/api` para o backend FastAPI;
- `/api/ws/*` para conexões WebSocket.

O arquivo `default.conf` é um rascunho funcional para a nova stack e não altera o Streamlit legado.
