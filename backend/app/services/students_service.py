import hashlib

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.students import ClassRead, StudentCreate, StudentRead, StudentUpdate
from app.schemas.users import CurrentUser


def generate_student_hash(name: str, matricula: str | None) -> str:
    base = f"{name}_{matricula or ''}".lower().strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def list_classes_for_user(db: Session, current_user: CurrentUser) -> list[ClassRead]:
    params: dict[str, object] = {}
    if current_user.role == "admin":
        query = """
            SELECT id, nome, COALESCE(identificador, '') AS identificador
            FROM classes
            ORDER BY nome, identificador
        """
    else:
        query = """
            SELECT DISTINCT c.id, c.nome, COALESCE(c.identificador, '') AS identificador
            FROM classes c
            JOIN teacher_subject_class tsc ON tsc.class_id = c.id
            WHERE tsc.teacher_id = :teacher_id
            ORDER BY c.nome, identificador
        """
        params["teacher_id"] = current_user.teacher_id

    rows = db.execute(text(query), params).mappings().all()
    return [ClassRead(**dict(row)) for row in rows]


def ensure_user_can_access_class(db: Session, current_user: CurrentUser, class_id: int | None) -> None:
    if class_id is None or current_user.role == "admin":
        return
    row = db.execute(
        text(
            """
            SELECT 1
            FROM teacher_subject_class
            WHERE teacher_id = :teacher_id
              AND class_id = :class_id
            LIMIT 1
            """
        ),
        {"teacher_id": current_user.teacher_id, "class_id": class_id},
    ).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Class not available for user")


def list_students_for_user(db: Session, current_user: CurrentUser, class_id: int | None = None) -> list[StudentRead]:
    params: dict[str, object] = {}
    conditions = ["s.ativo = TRUE"]

    if class_id is not None:
        conditions.append("s.class_id = :class_id")
        params["class_id"] = class_id

    if current_user.role == "professor":
        conditions.append(
            """
            s.class_id IN (
                SELECT class_id
                FROM teacher_subject_class
                WHERE teacher_id = :teacher_id
            )
            """
        )
        params["teacher_id"] = current_user.teacher_id

    rows = db.execute(
        text(
            f"""
            SELECT
                s.id,
                s.name,
                s.matricula,
                s.class_id,
                c.nome AS class_name,
                COALESCE(c.identificador, '') AS class_identifier
            FROM students s
            LEFT JOIN classes c ON c.id = s.class_id
            WHERE {" AND ".join(conditions)}
            ORDER BY s.name
            """
        ),
        params,
    ).mappings().all()

    return [StudentRead(**dict(row)) for row in rows]


def get_student_for_user(db: Session, current_user: CurrentUser, student_id: str) -> StudentRead:
    params: dict[str, object] = {"student_id": student_id}
    conditions = ["s.id = :student_id"]
    if current_user.role == "professor":
        conditions.append(
            """
            s.class_id IN (
                SELECT class_id
                FROM teacher_subject_class
                WHERE teacher_id = :teacher_id
            )
            """
        )
        params["teacher_id"] = current_user.teacher_id

    row = db.execute(
        text(
            f"""
            SELECT
                s.id,
                s.name,
                s.matricula,
                s.class_id,
                c.nome AS class_name,
                COALESCE(c.identificador, '') AS class_identifier
            FROM students s
            LEFT JOIN classes c ON c.id = s.class_id
            WHERE {" AND ".join(conditions)}
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return StudentRead(**dict(row))


def create_student(db: Session, current_user: CurrentUser, payload: StudentCreate) -> StudentRead:
    name = payload.name.strip()
    matricula = (payload.matricula or "").strip() or None
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Student name is required")

    ensure_user_can_access_class(db, current_user, payload.class_id)
    student_id = generate_student_hash(name, matricula)
    effective_student_id = student_id

    if matricula and payload.class_id is not None:
        existing = db.execute(
            text(
                """
                SELECT id
                FROM students
                WHERE class_id = :class_id
                  AND UPPER(BTRIM(COALESCE(matricula, ''))) = UPPER(BTRIM(:matricula))
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """
            ),
            {"class_id": payload.class_id, "matricula": matricula},
        ).mappings().first()
        if existing:
            effective_student_id = existing["id"]

    db.execute(
        text(
            """
            INSERT INTO students (id, name, matricula, class_id, ativo)
            VALUES (:id, :name, :matricula, :class_id, TRUE)
            ON CONFLICT (id)
            DO UPDATE SET
                name = EXCLUDED.name,
                matricula = COALESCE(EXCLUDED.matricula, students.matricula),
                class_id = COALESCE(EXCLUDED.class_id, students.class_id),
                ativo = TRUE
            """
        ),
        {
            "id": effective_student_id,
            "name": name,
            "matricula": matricula,
            "class_id": payload.class_id,
        },
    )
    db.commit()
    return get_student_for_user(db, current_user, effective_student_id)


def update_student(db: Session, current_user: CurrentUser, student_id: str, payload: StudentUpdate) -> StudentRead:
    current = get_student_for_user(db, current_user, student_id)
    target_class_id = payload.class_id if payload.class_id is not None else current.class_id
    ensure_user_can_access_class(db, current_user, target_class_id)

    row = db.execute(
        text(
            """
            UPDATE students
            SET
                name = COALESCE(:name, name),
                matricula = COALESCE(:matricula, matricula),
                class_id = COALESCE(:class_id, class_id),
                ativo = COALESCE(:ativo, ativo)
            WHERE id = :student_id
            RETURNING id
            """
        ),
        {
            "student_id": student_id,
            "name": payload.name.strip() if payload.name else None,
            "matricula": payload.matricula.strip() if payload.matricula else None,
            "class_id": payload.class_id,
            "ativo": payload.ativo,
        },
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    db.commit()
    return get_student_for_user(db, current_user, student_id)


def deactivate_student(db: Session, current_user: CurrentUser, student_id: str) -> None:
    get_student_for_user(db, current_user, student_id)
    db.execute(text("UPDATE students SET ativo = FALSE WHERE id = :student_id"), {"student_id": student_id})
    db.commit()


def get_next_student_registration(db: Session, prefix: str, min_digits: int = 3) -> str:
    normalized_prefix = prefix.strip().upper()
    rows = db.execute(
        text(
            """
            SELECT matricula
            FROM students
            WHERE UPPER(COALESCE(matricula, '')) LIKE :prefix
            ORDER BY created_at DESC NULLS LAST, matricula DESC
            """
        ),
        {"prefix": f"{normalized_prefix}%"},
    ).all()

    max_sequence = 0
    for (matricula,) in rows:
        value = str(matricula or "").strip().upper()
        if not value.startswith(normalized_prefix):
            continue
        suffix = value[len(normalized_prefix):]
        if suffix.isdigit() and len(suffix) >= min_digits:
            max_sequence = max(max_sequence, int(suffix))
    return f"{normalized_prefix}{str(max_sequence + 1).zfill(min_digits)}"
