from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.monitoring_service import runtime_monitoring_service
from app.services.realtime_event_service import monitoring_event_hub

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/monitoring")
async def monitoring_socket(websocket: WebSocket) -> None:
    await monitoring_event_hub.connect(websocket)
    await websocket.send_json(
        {
            "type": "connection.ready",
            "payload": runtime_monitoring_service.get_status().model_dump(mode="json"),
        }
    )
    try:
        while True:
            message = await websocket.receive_json()
            await websocket.send_json(
                {
                    "type": "connection.echo",
                    "payload": {
                        "message": message,
                        "status": runtime_monitoring_service.get_status().model_dump(mode="json"),
                    },
                }
            )
    except WebSocketDisconnect:
        monitoring_event_hub.disconnect(websocket)
        return
