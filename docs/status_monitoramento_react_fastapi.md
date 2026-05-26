# Status do monitoramento React + FastAPI

Atualizado em 2026-05-26.

## Arquivos criados

- `backend/app/services/rtsp_capture_service.py`
- `backend/app/services/realtime_event_service.py`
- `frontend/src/hooks/useMonitoring.ts`
- `frontend/src/components/monitoring/MonitoringVideoPanel.tsx`
- `frontend/src/components/monitoring/MonitoringStatusPanel.tsx`
- `frontend/src/components/monitoring/RealtimeEventsList.tsx`
- `docs/status_monitoramento_react_fastapi.md`

## Arquivos alterados

- `backend/app/api/routes_monitoring.py`
- `backend/app/core/config.py`
- `backend/app/main.py`
- `backend/app/schemas/monitoring.py`
- `backend/app/services/monitoring_service.py`
- `backend/app/websocket/monitoring.py`
- `frontend/src/features/monitoring/monitoringApi.ts`
- `frontend/src/pages/Monitoring.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/services/websocket.ts`
- `frontend/src/types/monitoring.ts`
- `frontend/vite.config.ts`

## Endpoints disponíveis

- `GET /api/monitoring/status`
- `POST /api/monitoring/start`
- `POST /api/monitoring/stop`
- `GET /api/monitoring/video-feed`
- `WS /ws/monitoring`
- `WS /api/ws/monitoring`

As rotas antigas de sessão foram mantidas:

- `GET /api/monitoring/options`
- `POST /api/monitoring/sessions`
- `POST /api/monitoring/sessions/{session_id}/close`
- `GET /api/monitoring/sessions/{session_id}`

## Como iniciar o backend

Na raiz do projeto:

```bash
PYTHONPATH=backend .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Antes de testar câmera real, defina `RTSP_URL` no `.env`, por exemplo:

```env
RTSP_URL=rtsp://usuario:senha@ip:554/Streaming/Channels/101
```

## Como iniciar o frontend

Na pasta `frontend`:

```bash
npm run dev
```

Neste ambiente, quando o `npm` global não estiver no PATH:

```bash
PATH=/home/starley/Documents/projects/insightface/.local/node/node-v22.22.3-linux-x64/bin:$PATH npm run dev -- --host 0.0.0.0
```

Variáveis suportadas no frontend:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Se essas variáveis não existirem, o frontend usa `/api` e o proxy do Vite encaminha para `localhost:8000`.

## Como testar RTSP

1. Configure `RTSP_URL` no `.env`.
2. Inicie o backend.
3. Inicie uma sessão:

```bash
curl -X POST http://localhost:8000/api/monitoring/start \
  -H "Authorization: Bearer SEU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"camera_id":1,"disciplina_id":1,"turma_id":1,"tipo_aula":"Exposição"}'
```

4. Consulte o status:

```bash
curl http://localhost:8000/api/monitoring/status
```

Estados esperados de câmera:

- `active`: OpenCV abriu o stream e está capturando frames.
- `unavailable`: OpenCV não abriu ou perdeu o stream.
- `reconnecting`: houve falha de leitura e o serviço tentará reconectar.
- `error`: configuração ausente ou OpenCV indisponível.
- `idle`: monitoramento parado.

## Como testar video-feed

Com uma sessão ativa:

```bash
curl -I http://localhost:8000/api/monitoring/video-feed
```

Ou abra no navegador:

```text
http://localhost:8000/api/monitoring/video-feed
```

No frontend, a página de Monitoramento usa:

```text
GET /api/monitoring/video-feed
```

via `<img>`, sem enviar RTSP direto ao React.

## Como testar WebSocket

Com o backend rodando:

```bash
npx wscat -c ws://localhost:8000/ws/monitoring
```

Também existe a rota prefixada:

```bash
npx wscat -c ws://localhost:8000/api/ws/monitoring
```

Ao iniciar monitoramento, o WebSocket publica eventos como:

```json
{
  "type": "behavior_event",
  "payload": {
    "session_id": 1,
    "student_name": "Aguardando identificação",
    "behavior": "Aguardando",
    "confidence": 0,
    "camera_status": "active",
    "websocket_status": "connected",
    "timestamp": "2026-05-26T..."
  }
}
```

## O que ainda falta para integrar YOLOv11-Pose

- Extrair do legado o carregamento do modelo `ultralytics.YOLO`.
- Criar adapter em `backend/app/ai/pose`.
- Enviar frames capturados pelo `RTSPCaptureService` para um worker de inferência.
- Transformar keypoints/poses em eventos leves para o `MonitoringService`.
- Manter throttling para não processar todos os frames quando o hardware não suportar.

## O que ainda falta para integrar InsightFace

- Extrair o carregamento `FaceAnalysis(name="buffalo_l")` do legado.
- Criar adapter em `backend/app/ai/face`.
- Reaproveitar embeddings já existentes em `data/embeddings.npy`, `data/names.pkl` e/ou `face_embeddings`.
- Associar reconhecimento facial ao escopo da turma/sessão.
- Publicar `student_name`, `confidence` e identificador do aluno nos eventos WebSocket.

## Próximos passos recomendados

1. Validar `RTSP_URL` real da câmera Hikvision ou relay.
2. Testar `video-feed` isolado antes de ativar IA.
3. Criar worker de IA separado do loop de captura.
4. Conectar YOLOv11-Pose mantendo a lógica legada como referência.
5. Conectar InsightFace usando adapters, sem duplicar regras.
6. Conectar `BehaviorEpisodeManager` para persistir episódios reais.
7. Adicionar testes de contrato para `start`, `stop`, `status` e WebSocket.

## Situação atual

A tela React de Monitoramento já consegue:

- consultar status ao abrir;
- conectar WebSocket;
- iniciar sessão via `POST /api/monitoring/start`;
- tentar abrir `RTSP_URL` no backend com OpenCV/FFmpeg;
- exibir o MJPEG via `<img>`;
- mostrar status de câmera, WebSocket, aluno, comportamento e confiança;
- listar eventos em tempo real;
- parar sessão via `POST /api/monitoring/stop`;
- liberar a câmera ao parar.

A IA ainda não está conectada nesta etapa; os eventos comportamentais são simulados e mantêm os campos que serão preenchidos por YOLOv11-Pose e InsightFace.
