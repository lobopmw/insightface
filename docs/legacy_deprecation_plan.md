# Plano de Desativação do Streamlit Legado

A autorização para remover código antigo será aplicada somente quando a funcionalidade equivalente estiver implementada, validada e documentada na nova stack.

## Não Remover Ainda

- `src/main.py`: ainda é o ponto de entrada funcional da aplicação atual.
- `src/insightface_classroom.py`: ainda contém runtime de IA, classificação comportamental, monitoramento e telas operacionais.
- `src/control_database_postgres.py`: ainda concentra schema e regras de acesso ao PostgreSQL atual.
- `src/register_face_multi_images_avg.py`: ainda contém fluxo de embeddings InsightFace/pgvector.
- `src/relay_rtsp_server.py` e `src/socket_video_stream.py`: ainda são referência para streaming RTSP/socket.
- `src/ui/*`: ainda contém telas administrativas e relatórios completos.
- `src/services/report_service.py` e `src/repositories/report_repository.py`: ainda têm regras mais completas que o resumo inicial do FastAPI.

## Pode Ser Removido Depois

Um arquivo legado só deve ser removido quando:

1. a nova API cobrir o mesmo comportamento;
2. a interface React cobrir o fluxo do usuário;
3. existir teste ou validação manual registrada;
4. a operação em produção não depender mais do Streamlit;
5. houver backup ou histórico Git limpo da versão antiga.

## Ordem Sugerida

1. Migrar autenticação e usuários administrativos.
2. Migrar cadastro de alunos e captura por poses.
3. Migrar relatórios pedagógicos completos.
4. Migrar runtime de monitoramento para serviço backend.
5. Migrar streaming RTSP/WebSocket.
6. Desativar Streamlit em ambiente de homologação.
7. Remover arquivos legados obsoletos.
