from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.monitoring import (
    MonitoringSessionClose,
    MonitoringSessionCreate,
    MonitoringSessionRead,
    MonitoringSessionSummary,
    MonitoringOptionsResponse,
    MonitoringStatusResponse,
)
from app.schemas.users import CurrentUser
from app.services.monitoring_service import (
    close_monitoring_session,
    create_monitoring_session,
    get_monitoring_session,
    list_monitoring_options,
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/status", response_model=MonitoringStatusResponse)
def monitoring_status() -> MonitoringStatusResponse:
    return MonitoringStatusResponse(status="idle")


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
