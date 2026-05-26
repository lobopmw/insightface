from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.reports import BehaviorEpisodeListResponse, ReportSummaryResponse
from app.schemas.users import CurrentUser
from app.services.reports_service import get_behavior_summary, list_behavior_episodes

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/summary", response_model=ReportSummaryResponse)
def reports_summary(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReportSummaryResponse:
    return ReportSummaryResponse(items=get_behavior_summary(db, current_user))


@router.get("/episodes", response_model=BehaviorEpisodeListResponse)
def reports_episodes(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    student_name: Annotated[str | None, Query()] = None,
    behavior: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BehaviorEpisodeListResponse:
    items, total = list_behavior_episodes(
        db,
        current_user,
        student_name=student_name,
        behavior=behavior,
        limit=limit,
        offset=offset,
    )
    return BehaviorEpisodeListResponse(items=items, total=total)
