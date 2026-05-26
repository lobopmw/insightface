from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/monitoring")
async def monitoring_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "connection.ready", "payload": {"status": "idle"}})
    try:
        while True:
            message = await websocket.receive_json()
            await websocket.send_json({"type": "echo", "payload": message})
    except WebSocketDisconnect:
        return
