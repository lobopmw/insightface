import csv
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.students import PoseFaceStatus, StudentFaceStatus, StudentRead

POSES = ("frontal", "lateral_esquerda", "lateral_direita", "cabeca_baixa")
MIN_IMAGES_PER_POSE = 10
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
STUDENTS_IMAGE_DIR = DATA_DIR / "alunos"
MAPPING_CSV = DATA_DIR / "mapeamento_alunos.csv"


def _student_dir(student_id: str) -> Path:
    return STUDENTS_IMAGE_DIR / student_id


def _pose_dir(student_id: str, pose: str) -> Path:
    return _student_dir(student_id) / pose


def ensure_student_mapping(student: StudentRead) -> None:
    if not student.id or not student.name:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if MAPPING_CSV.exists():
        with MAPPING_CSV.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = [dict(row) for row in reader]

    next_row = {"nome": student.name, "matricula": student.matricula or "", "hash": student.id}
    updated = False
    for row in rows:
        if str(row.get("hash", "")).strip() == student.id:
            row.update(next_row)
            updated = True
            break
    if not updated:
        rows.append(next_row)

    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        row_hash = str(row.get("hash", "")).strip()
        if row_hash:
            deduped[row_hash] = {
                "nome": str(row.get("nome", "")).strip(),
                "matricula": str(row.get("matricula", "")).strip(),
                "hash": row_hash,
            }

    with MAPPING_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["nome", "matricula", "hash"])
        writer.writeheader()
        writer.writerows(deduped.values())


def get_face_status(db: Session, student: StudentRead) -> StudentFaceStatus:
    pose_status = {}
    for pose in POSES:
        directory = _pose_dir(student.id, pose)
        count = 0
        if directory.exists():
            count = sum(1 for item in directory.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"})
        pose_status[pose] = PoseFaceStatus(count=count, complete=count >= MIN_IMAGES_PER_POSE)

    embedding_row = db.execute(
        text("SELECT 1 FROM face_embeddings WHERE student_hash = :student_id LIMIT 1"),
        {"student_id": student.id},
    ).first()

    return StudentFaceStatus(
        student_id=student.id,
        frontal=pose_status["frontal"],
        lateral_esquerda=pose_status["lateral_esquerda"],
        lateral_direita=pose_status["lateral_direita"],
        cabeca_baixa=pose_status["cabeca_baixa"],
        embeddings_generated=embedding_row is not None,
    )


async def save_face_images(
    db: Session,
    student: StudentRead,
    pose: str,
    files: list[UploadFile],
    start_index: int | None = None,
) -> tuple[int, StudentFaceStatus]:
    if pose not in POSES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid pose")
    if not files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="At least one image is required")

    ensure_student_mapping(student)
    directory = _pose_dir(student.id, pose)
    directory.mkdir(parents=True, exist_ok=True)

    saved = 0
    for index, upload in enumerate(files, start=start_index or 1):
        content = await upload.read()
        if not content:
            continue
        content_type = (upload.content_type or "").lower()
        if content_type and not content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only image uploads are accepted")
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix and suffix not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only jpg, jpeg and png are accepted")
        filename = f"{pose}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{index}.jpg"
        (directory / filename).write_bytes(content)
        saved += 1

    return saved, get_face_status(db, student)
