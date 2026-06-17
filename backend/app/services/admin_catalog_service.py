from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

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
    TeacherRead,
)
from app.schemas.students import StudentRead, StudentUpdate
from app.schemas.users import CurrentUser
from app.services.admin_users_service import ensure_admin
from app.services.students_service import list_students_for_user, update_student


def _sync_missing_teacher_profiles(db: Session) -> None:
    db.execute(
        text(
            """
            INSERT INTO teachers (user_id, nome)
            SELECT u.id, u.nome
            FROM users u
            LEFT JOIN teachers t ON t.user_id = u.id
            WHERE LOWER(COALESCE(u.role, 'professor')) = 'professor'
              AND u.ativo = TRUE
              AND t.id IS NULL
            ON CONFLICT (user_id)
            DO UPDATE SET nome = EXCLUDED.nome
            """
        )
    )
    db.execute(
        text(
            """
            UPDATE teachers t
            SET nome = u.nome
            FROM users u
            WHERE u.id = t.user_id
              AND LOWER(COALESCE(u.role, 'professor')) = 'professor'
              AND t.nome IS DISTINCT FROM u.nome
            """
        )
    )
    db.commit()


def _list_subjects(db: Session) -> list[SubjectRead]:
    rows = db.execute(text("SELECT id, nome FROM subjects ORDER BY nome")).mappings().all()
    return [SubjectRead(**dict(row)) for row in rows]


def _list_classes(db: Session) -> list[ClassRead]:
    rows = db.execute(
        text("SELECT id, nome, COALESCE(identificador, '') AS identificador FROM classes ORDER BY nome, identificador")
    ).mappings().all()
    return [ClassRead(**dict(row)) for row in rows]


def _list_teachers(db: Session) -> list[TeacherRead]:
    _sync_missing_teacher_profiles(db)
    rows = db.execute(
        text(
            """
            SELECT t.id, t.user_id, t.nome, u.cpf, u.email, u.ativo
            FROM teachers t
            JOIN users u ON u.id = t.user_id
            WHERE LOWER(COALESCE(u.role, 'professor')) = 'professor'
            ORDER BY t.nome
            """
        )
    ).mappings().all()
    return [TeacherRead(**dict(row)) for row in rows]


def _list_assignments(db: Session) -> list[AssignmentRead]:
    rows = db.execute(
        text(
            """
            SELECT
                tsc.id,
                tsc.teacher_id,
                t.nome AS teacher_name,
                tsc.subject_id,
                s.nome AS subject_name,
                tsc.class_id,
                c.nome AS class_name,
                COALESCE(c.identificador, '') AS class_identifier
            FROM teacher_subject_class tsc
            JOIN teachers t ON t.id = tsc.teacher_id
            JOIN subjects s ON s.id = tsc.subject_id
            JOIN classes c ON c.id = tsc.class_id
            ORDER BY t.nome, s.nome, c.nome, c.identificador
            """
        )
    ).mappings().all()
    return [AssignmentRead(**dict(row)) for row in rows]


def get_catalog(db: Session, current_user: CurrentUser) -> CatalogResponse:
    ensure_admin(current_user)
    return CatalogResponse(
        subjects=_list_subjects(db),
        classes=_list_classes(db),
        teachers=_list_teachers(db),
        assignments=_list_assignments(db),
    )


def create_subject(db: Session, current_user: CurrentUser, payload: SubjectCreate) -> SubjectRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            INSERT INTO subjects (nome)
            VALUES (:nome)
            ON CONFLICT (nome) DO UPDATE SET nome = EXCLUDED.nome
            RETURNING id, nome
            """
        ),
        {"nome": payload.nome.strip()},
    ).mappings().one()
    db.commit()
    return SubjectRead(**dict(row))


def update_subject(db: Session, current_user: CurrentUser, subject_id: int, payload: SubjectUpdate) -> SubjectRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            UPDATE subjects
            SET nome = COALESCE(:nome, nome)
            WHERE id = :subject_id
            RETURNING id, nome
            """
        ),
        {"subject_id": subject_id, "nome": payload.nome.strip() if payload.nome else None},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subject not found")
    db.commit()
    return SubjectRead(**dict(row))


def delete_subject(db: Session, current_user: CurrentUser, subject_id: int) -> None:
    ensure_admin(current_user)
    try:
        result = db.execute(text("DELETE FROM subjects WHERE id = :subject_id"), {"subject_id": subject_id})
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Subject is in use") from exc
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subject not found")


def create_class(db: Session, current_user: CurrentUser, payload: ClassCreate) -> ClassRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            INSERT INTO classes (nome, identificador)
            VALUES (:nome, :identificador)
            ON CONFLICT (nome, identificador)
            DO UPDATE SET nome = EXCLUDED.nome
            RETURNING id, nome, COALESCE(identificador, '') AS identificador
            """
        ),
        {"nome": payload.nome.strip(), "identificador": payload.identificador.strip() if payload.identificador else None},
    ).mappings().one()
    db.commit()
    return ClassRead(**dict(row))


def update_class(db: Session, current_user: CurrentUser, class_id: int, payload: ClassUpdate) -> ClassRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            UPDATE classes
            SET
                nome = COALESCE(:nome, nome),
                identificador = COALESCE(:identificador, identificador)
            WHERE id = :class_id
            RETURNING id, nome, COALESCE(identificador, '') AS identificador
            """
        ),
        {
            "class_id": class_id,
            "nome": payload.nome.strip() if payload.nome else None,
            "identificador": payload.identificador.strip() if payload.identificador else None,
        },
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")
    db.commit()
    return ClassRead(**dict(row))


def delete_class(db: Session, current_user: CurrentUser, class_id: int) -> None:
    ensure_admin(current_user)
    try:
        result = db.execute(text("DELETE FROM classes WHERE id = :class_id"), {"class_id": class_id})
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Class is in use") from exc
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")


def create_assignment(db: Session, current_user: CurrentUser, payload: AssignmentCreate) -> AssignmentRead:
    ensure_admin(current_user)
    row = db.execute(
        text(
            """
            INSERT INTO teacher_subject_class (teacher_id, subject_id, class_id)
            VALUES (:teacher_id, :subject_id, :class_id)
            ON CONFLICT (teacher_id, subject_id, class_id)
            DO UPDATE SET teacher_id = EXCLUDED.teacher_id
            RETURNING id
            """
        ),
        {"teacher_id": payload.teacher_id, "subject_id": payload.subject_id, "class_id": payload.class_id},
    ).mappings().one()
    db.commit()
    assignment = db.execute(
        text(
            """
            SELECT
                tsc.id,
                tsc.teacher_id,
                t.nome AS teacher_name,
                tsc.subject_id,
                s.nome AS subject_name,
                tsc.class_id,
                c.nome AS class_name,
                COALESCE(c.identificador, '') AS class_identifier
            FROM teacher_subject_class tsc
            JOIN teachers t ON t.id = tsc.teacher_id
            JOIN subjects s ON s.id = tsc.subject_id
            JOIN classes c ON c.id = tsc.class_id
            WHERE tsc.id = :assignment_id
            """
        ),
        {"assignment_id": row["id"]},
    ).mappings().one()
    return AssignmentRead(**dict(assignment))


def delete_assignment(db: Session, current_user: CurrentUser, assignment_id: int) -> None:
    ensure_admin(current_user)
    result = db.execute(text("DELETE FROM teacher_subject_class WHERE id = :assignment_id"), {"assignment_id": assignment_id})
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")


def list_admin_students(db: Session, current_user: CurrentUser) -> list[StudentRead]:
    ensure_admin(current_user)
    return list_students_for_user(db, current_user)


def update_admin_student(db: Session, current_user: CurrentUser, student_id: str, payload: StudentUpdate) -> StudentRead:
    ensure_admin(current_user)
    return update_student(db, current_user, student_id, payload)
