from fastapi import WebSocket


class MonitoringEventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self.last_event: str | None = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def publish(self, event_type: str, payload: dict) -> None:
        self.last_event = event_type
        disconnected: list[WebSocket] = []
        message = {"type": event_type, "payload": payload}
        for websocket in list(self._clients):
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        for websocket in disconnected:
            self.disconnect(websocket)


monitoring_event_hub = MonitoringEventHub()
