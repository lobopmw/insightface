from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.users import CurrentUser
from app.services.auth_service import authenticate_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    return authenticate_user(db, payload)


@router.get("/me", response_model=CurrentUser)
def read_current_user(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    return current_user
