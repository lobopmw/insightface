from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.admin_users import AdminUserCreate, AdminUserListResponse, AdminUserRead, AdminUserUpdate
from app.schemas.users import CurrentUser
from app.services.admin_users_service import create_user, list_users, update_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=AdminUserListResponse)
def list_users_route(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AdminUserListResponse:
    return AdminUserListResponse(items=list_users(db, current_user))


@router.post("", response_model=AdminUserRead, status_code=status.HTTP_201_CREATED)
def create_user_route(
    payload: AdminUserCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AdminUserRead:
    return create_user(db, current_user, payload)


@router.patch("/{user_id}", response_model=AdminUserRead)
def update_user_route(
    user_id: int,
    payload: AdminUserUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AdminUserRead:
    return update_user(db, current_user, user_id, payload)
