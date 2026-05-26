import sys
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.students import EmbeddingGenerationResponse, StudentRead
from app.services.face_capture_service import ensure_student_mapping, get_face_status

ROOT_DIR = Path(__file__).resolve().parents[3]
LEGACY_SRC_DIR = ROOT_DIR / "src"


def generate_embeddings_for_student(db: Session, student: StudentRead) -> EmbeddingGenerationResponse:
    ensure_student_mapping(student)

    if str(LEGACY_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_SRC_DIR))

    try:
        from register_face_multi_images_avg import generate_student_embedding
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"InsightFace runtime unavailable: {exc}",
        ) from exc

    success, message = generate_student_embedding(student.id, student.name, student.matricula or None)
    return EmbeddingGenerationResponse(
        student_id=student.id,
        success=bool(success),
        message=str(message),
        status=get_face_status(db, student),
    )
