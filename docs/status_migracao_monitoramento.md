# Status da migracao do monitoramento em tempo real

Analise realizada em 2026-05-26 no projeto local. Este documento descreve o estado atual da migracao do fluxo de monitoramento do legado Streamlit para a nova stack React + FastAPI, sem alterar codigo de IA, banco ou aplicacao.

## Estrutura atual do projeto

- `frontend/`: aplicacao React + Vite + TypeScript, com Tailwind, React Query e Zustand.
- `backend/`: API FastAPI nova, com rotas REST, WebSocket inicial, servicos, schemas, configuracao, banco e migrations Alembic.
- `src/`: aplicacao legada Streamlit e scripts operacionais ainda funcionais, incluindo monitoramento, IA, RTSP/socket e integracao direta com PostgreSQL.
- `data/`: imagens de alunos, embeddings locais, nomes e arquivos CSV simulados.
- `docker/`: compose da nova stack com PostgreSQL, backend, frontend e inicializacao admin.
- `nginx/`: configuracao de proxy para frontend/API/WebSocket.
- `docs/`: inventarios e planos de migracao existentes.
- `legacy/`: artefatos de migracao SQLite/PostgreSQL antigos.
- `sql/`: scripts auxiliares de seed/reset de dados.

## Arquivos do frontend React ja existentes

Arquivos diretamente ligados ao monitoramento:

- `frontend/src/pages/Monitoring.tsx`: tela de monitoramento, selecao de disciplina/turma/tipo de aula, botoes `Iniciar` e `Encerrar`, e placeholder do video.
- `frontend/src/components/monitoring/MonitoringPanel.tsx`: painel lateral com opcoes da sessao, status, id da sessao e indicador `RTSP pendente`.
- `frontend/src/features/monitoring/monitoringApi.ts`: chamadas REST para opcoes, criacao de sessao e encerramento de sessao.
- `frontend/src/hooks/useMonitoringOptions.ts`: carrega opcoes de monitoramento via React Query.
- `frontend/src/stores/monitoringStore.ts`: guarda `activeSession`, estado de loading/erro e executa start/stop REST.
- `frontend/src/types/monitoring.ts`: tipos TypeScript para opcoes e sessao.
- `frontend/src/services/websocket.ts`: helper para abrir `WS /api/ws/monitoring`, ainda sem uso na tela de monitoramento.
- `frontend/src/pages/Settings.tsx`: mostra a base URL do WebSocket.

Arquivos de suporte relevantes:

- `frontend/src/services/api.ts`: cliente HTTP com base `VITE_API_BASE_URL` ou `/api`.
- `frontend/src/stores/authStore.ts`: token de autenticacao usado pelas chamadas protegidas.
- `frontend/src/app/App.tsx`, `frontend/src/app/main.tsx`, `frontend/src/components/layout/AppLayout.tsx`: estrutura geral da aplicacao.

## Arquivos do backend FastAPI ja existentes

Arquivos diretamente ligados ao monitoramento:

- `backend/app/api/routes_monitoring.py`: expoe `GET /api/monitoring/status`, `GET /api/monitoring/options`, `POST /api/monitoring/sessions`, `GET /api/monitoring/sessions/{id}` e `POST /api/monitoring/sessions/{id}/close`.
- `backend/app/services/monitoring_service.py`: lista disciplinas/turmas permitidas, valida escopo do professor, cria sessao em `monitoring_sessions`, encerra sessao e consulta resumo basico.
- `backend/app/schemas/monitoring.py`: contratos Pydantic de status, opcoes e sessao.
- `backend/app/websocket/monitoring.py`: WebSocket inicial em `/api/ws/monitoring`, aceita conexao, envia `connection.ready` e responde mensagens com `echo`.
- `backend/app/services/behavior_service.py`: funcao preparada para inserir episodios em `behavior_episode`.
- `backend/app/ai/behavior/episode_manager.py`: gerenciador de episodios comportamentais migrado para o backend.
- `backend/app/ai/face/__init__.py`, `backend/app/ai/pose/__init__.py`, `backend/app/ai/streaming/__init__.py`: pacotes placeholder para futuras extracoes de InsightFace, YOLO e streaming.

Arquivos de infraestrutura relevantes:

- `backend/app/main.py`: registra rotas REST e WebSocket sob `API_PREFIX`.
- `backend/app/core/config.py`: configuracao de API, CORS, banco e seguranca.
- `backend/app/db/session.py`: sessao SQLAlchemy para PostgreSQL.
- `backend/migrations/versions/20260526_0001_initial_schema.py`: schema com `monitoring_sessions`, `behavior_episode`, `face_embeddings`, `students`, `classes`, `subjects`, `teachers` e demais tabelas.
- `backend/requirements.txt`: dependencias web/banco; dependencias de IA permanecem comentadas como preservadas no legado.

## O que ja esta pronto

- A nova tela React de monitoramento existe e permite selecionar disciplina, turma e tipo de aula.
- O frontend chama o backend para carregar opcoes de monitoramento.
- O frontend chama o backend para criar uma sessao.
- O frontend chama o backend para encerrar a sessao ativa.
- O backend cria registros reais em `monitoring_sessions` com `teacher_id`, `subject_id`, `class_id`, `lesson_type`, `session_date`, `start_time` e status `em_andamento`.
- O backend encerra registros em `monitoring_sessions` preenchendo `end_time` e status recebido, normalmente `encerrada`.
- O backend valida se um professor pode monitorar a combinacao disciplina/turma solicitada.
- A migration nova contempla as tabelas necessarias para sessoes, embeddings e episodios.
- A funcao `insert_behavior_episode` ja existe no backend e grava em `behavior_episode`.
- O `BehaviorEpisodeManager` ja foi copiado para `backend/app/ai/behavior/episode_manager.py`.
- O WebSocket esta registrado no FastAPI e responde conexoes basicas.
- O legado Streamlit ainda contem o fluxo funcional completo de monitoramento, com RTSP/socket, YOLO Pose, InsightFace, suavizacao de comportamento, episodios e persistencia.

## O que esta incompleto

- A nova tela React ainda nao exibe video real; mostra apenas placeholder.
- A nova tela React nao usa `createMonitoringSocket`.
- O WebSocket do backend nao envia frames, eventos de deteccao, status de camera, status de IA, overlays ou metricas de sessao.
- O endpoint `GET /api/monitoring/status` retorna sempre `idle`; nao consulta runtime real.
- Criar uma sessao no FastAPI nao inicia captura RTSP, nao conecta no relay, nao carrega modelos e nao inicia worker de IA.
- Encerrar uma sessao no FastAPI nao faz flush do `BehaviorEpisodeManager`, porque o manager ainda nao esta conectado ao runtime moderno.
- Os pacotes `backend/app/ai/face`, `backend/app/ai/pose` e `backend/app/ai/streaming` ainda nao contem adapters reais.
- As dependencias de IA nao estao no `backend/requirements.txt`; permanecem no fluxo legado.
- O compose novo (`docker/compose.yml`) nao possui servico `relay`; apenas define `RELAY_HOST` e `RELAY_PORT` no backend.
- Nao ha orquestrador de sessao em memoria/processo no backend para controlar camera, workers e clientes conectados.
- Nao ha contrato frontend/backend para mensagens de tempo real alem do `echo`.

## O que falta para iniciar/parar monitoramento

Para iniciar monitoramento real na nova stack, ainda falta:

- Criar um servico de runtime no backend para gerenciar o ciclo de vida da sessao ativa.
- Ao chamar `POST /api/monitoring/sessions`, alem de gravar a sessao, iniciar ou associar:
  - fonte de video RTSP/relay;
  - worker de inferencia YOLO Pose;
  - worker de reconhecimento InsightFace;
  - `BehaviorEpisodeManager`;
  - callback de persistencia em `behavior_episode`;
  - broadcaster WebSocket para status/eventos/video.
- Definir se o backend consumira diretamente RTSP, o socket do relay legado ou MJPEG do relay.
- Adicionar controle de concorrencia: uma sessao por professor, por turma, por camera ou global.
- Guardar estado operacional da sessao: `starting`, `running`, `stopping`, `stopped`, `error`.
- Expor erros operacionais legiveis: RTSP indisponivel, relay sem frame, modelo indisponivel, banco indisponivel.

Para parar monitoramento real na nova stack, ainda falta:

- No encerramento, chamar `flush_all` do `BehaviorEpisodeManager`.
- Parar worker de video e worker de IA.
- Fechar conexoes com relay/camera quando aplicavel.
- Encerrar/broadcastar status no WebSocket.
- Atualizar a sessao em `monitoring_sessions` somente depois de finalizar o flush, ou registrar falha parcial quando necessario.
- Limpar estado em memoria da sessao encerrada.

## Situacao do WebSocket

- Backend: existe `backend/app/websocket/monitoring.py`.
- Rota: `WS /api/ws/monitoring`.
- Comportamento atual: aceita conexao, envia uma mensagem `connection.ready` com `status: idle` e devolve mensagens recebidas como `echo`.
- Autenticacao: o frontend inclui `token` na query string, mas o backend ainda nao valida esse token no WebSocket.
- Uso no frontend: existe helper em `frontend/src/services/websocket.ts`, mas a tela `Monitoring.tsx` nao abre socket nem consome mensagens.
- Lacuna principal: nao ha protocolo de mensagens para `session.status`, `camera.status`, `frame`, `detections`, `behavior.updated`, `episode.persisted` ou `error`.

## Situacao do RTSP

- O RTSP funcional esta no legado:
  - `src/relay_rtsp_server.py` abre RTSP via OpenCV/FFMPEG, reconecta em falhas, serve frames por TCP e stream MJPEG.
  - `src/socket_video_stream.py` consome frames do relay por socket TCP.
  - `src/insightface_classroom.py` usa `VideoStream((RELAY_HOST, RELAY_PORT)).start()`.
- O `docker-compose.yml` legado possui servico `relay` com `RTSP_URL`.
- O compose novo em `docker/compose.yml` nao sobe relay; apenas passa `RELAY_HOST` e `RELAY_PORT` para o backend.
- O backend FastAPI novo nao possui cliente de RTSP/relay implementado.
- A interface React mostra explicitamente `RTSP pendente`.

## Situacao da integracao YOLO/InsightFace

- A integracao real ainda esta em `src/insightface_classroom.py`.
- YOLO:
  - usa `ultralytics.YOLO`;
  - escolhe `yolo11m-pose.pt` em CUDA e `yolo11n-pose.pt` em CPU;
  - roda dentro de `DetectorWorker`.
- InsightFace:
  - usa `insightface.app.FaceAnalysis` com modelo `buffalo_l`;
  - usa providers CUDA/CPU quando disponiveis;
  - carrega embeddings via `load_insightface_data`.
- O legado combina YOLO + InsightFace em background, sempre processando o frame mais recente.
- No backend novo, existem apenas namespaces vazios/placeholder para face, pose e streaming.
- `backend/requirements.txt` nao instala `opencv-python`, `ultralytics`, `insightface`, `onnxruntime-gpu` ou `torch`; ha apenas comentario indicando que dependencias de IA permanecem no legado ate a extracao.
- Portanto, a IA ainda nao foi conectada ao FastAPI.

## Situacao da gravacao no PostgreSQL

- A migration nova cria/garante tabelas relevantes:
  - `monitoring_sessions`;
  - `behavior_episode`;
  - `face_embeddings`;
  - `students`;
  - `teachers`;
  - `subjects`;
  - `classes`;
  - `teacher_subject_class`.
- A criacao e encerramento de sessoes ja gravam no PostgreSQL pela nova API.
- A funcao `backend/app/services/behavior_service.py::insert_behavior_episode` ja esta pronta para inserir episodios.
- O `BehaviorEpisodeManager` do backend aceita `persist_callback`, mas ainda nao esta instanciado por uma sessao FastAPI real.
- Na pratica, a nova stack grava `monitoring_sessions`, mas ainda nao grava episodios reais de IA em `behavior_episode`.
- A gravacao completa de episodios reais continua funcionando apenas no legado Streamlit, onde `BehaviorEpisodeManager` chama `insert_behavior_episode` durante o monitoramento e no encerramento.

## Proximos passos recomendados

1. Definir arquitetura do runtime moderno: processo unico FastAPI, worker separado ou servico dedicado para IA/streaming.
2. Migrar o cliente de streaming do legado para `backend/app/ai/streaming`, mantendo compatibilidade inicial com `src/relay_rtsp_server.py`.
3. Adicionar um servico de orquestracao de monitoramento no backend para iniciar/parar runtime por sessao.
4. Migrar adapters de YOLO e InsightFace para `backend/app/ai/pose` e `backend/app/ai/face`, sem alterar o comportamento dos modelos inicialmente.
5. Conectar `BehaviorEpisodeManager` ao runtime moderno e ao `insert_behavior_episode`.
6. Implementar protocolo WebSocket versionado para status, eventos e, se decidido, frames/overlays.
7. Ligar a tela React ao WebSocket para mostrar estado real da camera/IA e substituir o placeholder por stream ou frames.
8. Decidir como servir video no React: MJPEG direto do relay, frames pelo WebSocket, endpoint HTTP de stream no backend ou WebRTC.
9. Adicionar validacao de token no WebSocket.
10. Atualizar Docker da nova stack para incluir relay ou documentar uso de relay externo.
11. Criar testes unitarios do orquestrador e testes de contrato do WebSocket.
12. Fazer uma validacao manual fim a fim: login, opcoes, iniciar sessao, receber frames/status, gravar episodio, encerrar e consultar relatorio.

## Conclusao

O monitoramento em tempo real ainda nao funciona na nova stack React + FastAPI. O que funciona hoje na stack nova e o controle administrativo da sessao no banco: listar opcoes, criar sessao e encerrar sessao. O funcionamento real de camera, RTSP, YOLO, InsightFace, overlays, classificacao comportamental e gravacao de episodios ainda esta concentrado no legado Streamlit em `src/insightface_classroom.py` e arquivos auxiliares.
