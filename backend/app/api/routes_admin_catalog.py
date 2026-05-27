from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.admin_catalog import (
    AssignmentCreate,
    AssignmentRead,
    CatalogResponse,
    ClassCreate,
    ClassRead,
    ClassUpdate,
    SubjectCreate,
    SubjectRead,
    SubjectUpdate,
)
from app.schemas.students import StudentListResponse, StudentRead, StudentUpdate
from app.schemas.users import CurrentUser
from app.services.admin_catalog_service import (
    create_assignment,
    create_class,
    create_subject,
    delete_assignment,
    delete_class,
    delete_subject,
    get_catalog,
    list_admin_students,
    update_admin_student,
    update_class,
    update_subject,
)

router = APIRouter(prefix="/admin/catalog", tags=["admin-catalog"])


@router.get("", response_model=CatalogResponse)
def get_catalog_route(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> CatalogResponse:
    return get_catalog(db, current_user)


@router.post("/subjects", response_model=SubjectRead, status_code=status.HTTP_201_CREATED)
def create_subject_route(
    payload: SubjectCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> SubjectRead:
    return create_subject(db, current_user, payload)


@router.patch("/subjects/{subject_id}", response_model=SubjectRead)
def update_subject_route(
    subject_id: int,
    payload: SubjectUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> SubjectRead:
    return update_subject(db, current_user, subject_id, payload)


@router.delete("/subjects/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject_route(
    subject_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    delete_subject(db, current_user, subject_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/classes", response_model=ClassRead, status_code=status.HTTP_201_CREATED)
def create_class_route(
    payload: ClassCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ClassRead:
    return create_class(db, current_user, payload)


@router.patch("/classes/{class_id}", response_model=ClassRead)
def update_class_route(
    class_id: int,
    payload: ClassUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ClassRead:
    return update_class(db, current_user, class_id, payload)


@router.delete("/classes/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_class_route(
    class_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    delete_class(db, current_user, class_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/assignments", response_model=AssignmentRead, status_code=status.HTTP_201_CREATED)
def create_assignment_route(
    payload: AssignmentCreate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AssignmentRead:
    return create_assignment(db, current_user, payload)


@router.delete("/assignments/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assignment_route(
    assignment_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    delete_assignment(db, current_user, assignment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/students", response_model=StudentListResponse)
def list_students_route(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StudentListResponse:
    return StudentListResponse(items=list_admin_students(db, current_user))


@router.patch("/students/{student_id}", response_model=StudentRead)
def update_student_route(
    student_id: str,
    payload: StudentUpdate,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StudentRead:
    return update_admin_student(db, current_user, student_id, payload)
