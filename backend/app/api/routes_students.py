from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.students import (
    ClassListResponse,
    NextRegistrationResponse,
    StudentCreate,
    StudentListResponse,
    StudentRead,
    StudentUpdate,
)
from app.schemas.users import CurrentUser
from app.services.students_service import (
    create_student,
    deactivate_student,
    get_next_student_registration,
    get_student_for_user,
    list_classes_for_user,
    list_students_for_user,
    update_student,
)

router = APIRouter(prefix="/students", tags=["students"])


@router.get("/classes", response_model=ClassListResponse)
def list_student_classes(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ClassListResponse:
    return ClassListResponse(items=list_classes_for_user(db, current_user))


@router.get("/next-registration", response_model=NextRegistrationResponse)
def next_registration(
    db: Annotated[Session, Depends(get_db)],
    prefix: Annotated[str | None, Query()] = None,
) -> NextRegistrationResponse:
    effective_prefix = prefix or f"MAT{datetime.now().year}"
    return NextRegistrationResponse(matricula=get_next_student_registration(db, effective_prefix))


@router.get("", response_model=StudentListResponse)
def list_students(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    class_id: Annotated[int | None, Query()] = None,
) -> StudentListResponse:
    return StudentListResponse(items=list_students_for_user(db, current_user, class_id=class_id))


@router.post("", response_model=StudentRead, status_code=status.HTTP_201_CREATED)
def create_student_route(
    payload: StudentCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StudentRead:
    return create_student(db, current_user, payload)


@router.get("/{student_id}", response_model=StudentRead)
def get_student_route(
    student_id: str,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StudentRead:
    return get_student_for_user(db, current_user, student_id)


@router.patch("/{student_id}", response_model=StudentRead)
def update_student_route(
    student_id: str,
    payload: StudentUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StudentRead:
    return update_student(db, current_user, student_id, payload)


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student_route(
    student_id: str,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    deactivate_student(db, current_user, student_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
