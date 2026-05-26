# Inventário Inicial da Migração

Esta etapa cria uma arquitetura paralela e preserva a aplicação Streamlit atual.

## Interface Streamlit

- `src/main.py`: shell da aplicação, login, sessão, cookies, cadastro de usuários e navegação.
- `src/insightface_classroom.py`: telas principais de cadastro, monitoramento, integração Streamlit/WebRTC e boa parte da lógica operacional.
- `src/capture_images_for_student.py`: fluxo Streamlit de captura por poses.
- `src/ui/admin_user_page.py`: tela administrativa de usuários.
- `src/ui/report_page.py`: tela Streamlit de relatórios.

## IA, Visão e Streaming

- `src/insightface_classroom.py`: runtime misto com OpenCV, YOLO, InsightFace, suavização de comportamento e renderização.
- `src/register_face_multi_images_avg.py`: geração e persistência de embeddings faciais com InsightFace/pgvector.
- `src/socket_video_stream.py`: cliente de frames por socket.
- `src/relay_rtsp_server.py`: relay RTSP Hikvision para baixa latência.
- `src/behavior_episode_service.py`: gerenciador reutilizável de episódios de comportamento.

## Banco, Autenticação e Permissões

- `src/control_database_postgres.py`: schema PostgreSQL/pgvector, usuários, turmas, disciplinas, sessões e consultas analíticas.
- `src/init_db.py`: inicialização do banco.
- `src/utils_criptografia.py`: hashing de identificação de alunos.
- `src/main.py`: autenticação Streamlit baseada em bcrypt, cookie e token assinado.

## Relatórios e Utilitários

- `src/services/report_service.py`: regras de agregação e preparação de dados de relatórios.
- `src/repositories/report_repository.py`: consultas de relatório via camada atual de banco.
- `src/utils/report_formatters.py`: formatação de relatório.
- `src/generate_monitoring_storage_guide_pdf.py` e `src/generate_report_page_guide_pdf.py`: geração de guias PDF.
- `sql/*.sql`: scripts de seed, reset e ajustes de dados.
- `legacy/sqlite_migration/*`: artefatos legados de migração SQLite para PostgreSQL.

## Direção de Extração

1. Isolar autenticação em `backend/app/api/routes_auth.py` usando a tabela `users` atual.
2. Migrar consultas de alunos, turmas, disciplinas e sessões para serviços FastAPI sem alterar schema.
3. Mover `BehaviorEpisodeManager` para `backend/app/ai/behavior` ou `backend/app/services`.
4. Encapsular câmera/relay em `backend/app/ai/streaming`.
5. Encapsular InsightFace e YOLO em serviços carregados sob demanda para suporte futuro a múltiplas câmeras.
6. Substituir telas Streamlit por páginas React conforme cada contrato de API for estabilizado.

## Extraído Nesta Rodada

- `BehaviorEpisodeManager` foi migrado para `backend/app/ai/behavior/episode_manager.py`.
- A persistência de episódios foi preparada em `backend/app/services/behavior_service.py`.
- O CRUD administrativo inicial de alunos foi implementado no backend FastAPI.
