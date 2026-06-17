from typing import Annotated

from fastapi import APIRouter, Body, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.monitoring import (
    MonitoringSessionClose,
    MonitoringSessionCreate,
    MonitoringSessionRead,
    MonitoringSessionSummary,
    MonitoringOptionsResponse,
    MonitoringStartRequest,
    MonitoringStatusResponse,
    MonitoringStopRequest,
)
from app.schemas.users import CurrentUser
from app.services.monitoring_service import (
    close_monitoring_session,
    create_monitoring_session,
    get_monitoring_session,
    list_monitoring_options,
    runtime_monitoring_service,
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/status", response_model=MonitoringStatusResponse)
def monitoring_status() -> MonitoringStatusResponse:
    return runtime_monitoring_service.get_status()


@router.get("/video-feed")
def monitoring_video_feed() -> StreamingResponse:
    return StreamingResponse(
        runtime_monitoring_service.video_feed(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/start", response_model=MonitoringStatusResponse)
async def start_monitoring(
    payload: MonitoringStartRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MonitoringStatusResponse:
    return await runtime_monitoring_service.start(db, current_user, payload)


@router.post("/stop", response_model=MonitoringStatusResponse)
async def stop_monitoring(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    payload: MonitoringStopRequest = Body(default_factory=MonitoringStopRequest),
) -> MonitoringStatusResponse:
    return await runtime_monitoring_service.stop(db, current_user, payload.status)


@router.get("/options", response_model=MonitoringOptionsResponse)
def monitoring_options(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MonitoringOptionsResponse:
    return list_monitoring_options(db, current_user)


@router.post("/sessions", response_model=MonitoringSessionRead, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: MonitoringSessionCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MonitoringSessionRead:
    return create_monitoring_session(db, current_user, payload)


@router.post("/sessions/{session_id}/close", response_model=MonitoringSessionRead)
def close_session(
    session_id: int,
    payload: MonitoringSessionClose,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MonitoringSessionRead:
    return close_monitoring_session(db, current_user, session_id, payload.status)


@router.get("/sessions/{session_id}", response_model=MonitoringSessionSummary)
def read_session(
    session_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MonitoringSessionSummary:
    return get_monitoring_session(db, current_user, session_id)
