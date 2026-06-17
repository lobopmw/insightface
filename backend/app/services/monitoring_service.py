import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas.monitoring import (
    MonitoringClass,
    MonitoringOptionsResponse,
    MonitoringSessionCreate,
    MonitoringSessionRead,
    MonitoringSessionSummary,
    MonitoringStartRequest,
    MonitoringStatusResponse,
    MonitoringSubject,
)
from app.schemas.users import CurrentUser
from app.services.face_recognition_service import FaceRecognitionResult, FaceRecognitionService
from app.services.pose_behavior_service import BehaviorTrack, PoseBehaviorService
from app.services.realtime_event_service import monitoring_event_hub
from app.services.rtsp_capture_service import RTSPCaptureService


LESSON_TYPES = ["Exposição", "Atividade", "Prova", "Revisão", "Outro"]
BEHAVIOR_SWITCH_CONFIRMATIONS = 3
DETECTION_HOLD_SECONDS = 10.0
IDENTITY_PROPAGATION_SECONDS = 8.0
IDENTITY_PROPAGATION_MIN_IOU = 0.12
IDENTITY_PROPAGATION_MAX_CENTER_DISTANCE = 0.45


class MonitoringService:
    def __init__(
        self,
        capture_service: RTSPCaptureService | None = None,
        face_recognition_service: FaceRecognitionService | None = None,
        pose_behavior_service: PoseBehaviorService | None = None,
    ):
        self.capture_service = capture_service or RTSPCaptureService()
        self.face_recognition_service = face_recognition_service or FaceRecognitionService()
        self.pose_behavior_service = pose_behavior_service or PoseBehaviorService()
        self.session_id: int | None = None
        self.status = "idle"
        self.error: str | None = None
        self.student_name = "Aguardando identificação"
        self.behavior = "Aguardando"
        self.confidence = 0.0
        self.timestamp: datetime | None = None
        self._event_task: asyncio.Task | None = None
        self._latest_detections: list[FaceRecognitionResult | BehaviorTrack] = []
        self._display_detection: FaceRecognitionResult | BehaviorTrack | None = None
        self._display_seen_at = 0.0
        self._display_candidate_key: tuple[str, str] | None = None
        self._display_candidate_count = 0

    async def start(
        self,
        db: Session,
        current_user: CurrentUser,
        payload: MonitoringStartRequest,
    ) -> MonitoringStatusResponse:
        if self.status in {"starting", "running"} and self.session_id is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Monitoring session already started")

        try:
            session_payload = payload.to_session_create()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

        self.status = "starting"
        self.error = None
        self.student_name = "Aguardando identificação"
        self.behavior = "Aguardando YOLO"
        self.confidence = 0.0
        self._latest_detections = []
        self._display_detection = None
        self._display_seen_at = 0.0
        self._display_candidate_key = None
        self._display_candidate_count = 0
        self.timestamp = datetime.now(UTC)
        await monitoring_event_hub.publish(
            "monitoring.starting",
            {
                "status": self.status,
                "session_id": None,
                "camera_id": payload.camera_id,
                "disciplina_id": session_payload.subject_id,
                "turma_id": session_payload.class_id,
                "tipo_aula": session_payload.lesson_type,
            },
        )

        session = create_monitoring_session(db, current_user, session_payload)
        self.session_id = session.id
        self.face_recognition_service.reload_embeddings()
        camera = self.capture_service.start()
        self.status = "running"
        recognition_error = self.face_recognition_service.load_error
        pose_error = self.pose_behavior_service.load_error
        if camera.error or recognition_error or pose_error:
            self.error = camera.error or recognition_error or pose_error
        self._start_event_task()

        await monitoring_event_hub.publish(
            "monitoring.started",
            {
                "status": self.status,
                "session_id": self.session_id,
                "camera_status": camera.status,
                "camera_source": camera.source,
                "camera_connected": camera.connected,
                "error": self.error,
            },
        )
        return self.get_status()

    async def stop(
        self,
        db: Session,
        current_user: CurrentUser,
        close_status: str = "encerrada",
    ) -> MonitoringStatusResponse:
        if self.session_id is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No active monitoring session")

        session_id = self.session_id
        self.status = "stopping"
        await monitoring_event_hub.publish("monitoring.stopping", {"status": self.status, "session_id": session_id})

        close_monitoring_session(db, current_user, session_id, close_status)
        await self._stop_event_task()
        camera = self.capture_service.stop()
        self.session_id = None
        self.status = "idle"
        self.error = None
        self.student_name = "Aguardando identificação"
        self.behavior = "Aguardando YOLO"
        self.confidence = 0.0
        self._latest_detections = []
        self._display_detection = None
        self._display_seen_at = 0.0
        self._display_candidate_key = None
        self._display_candidate_count = 0
        self.timestamp = datetime.now(UTC)

        await monitoring_event_hub.publish(
            "monitoring.stopped",
            {
                "status": self.status,
                "session_id": session_id,
                "camera_status": camera.status,
            },
        )
        return self.get_status()

    def get_status(self) -> MonitoringStatusResponse:
        camera = self.capture_service.get_status()
        self.timestamp = self.timestamp or datetime.now(UTC)
        return MonitoringStatusResponse(
            status=self.status,
            session_id=self.session_id,
            camera_status=camera.status,
            websocket_clients=monitoring_event_hub.client_count,
            last_event=monitoring_event_hub.last_event,
            error=self.error or camera.error,
            student_name=self.student_name,
            behavior=self.behavior,
            confidence=self.confidence,
            timestamp=self.timestamp,
        )

    def _start_event_task(self) -> None:
        if self._event_task is None or self._event_task.done():
            self._event_task = asyncio.create_task(self._publish_recognition_events())

    async def _stop_event_task(self) -> None:
        if self._event_task is None:
            return
        self._event_task.cancel()
        try:
            await self._event_task
        except asyncio.CancelledError:
            pass
        self._event_task = None

    async def _publish_recognition_events(self) -> None:
        while self.session_id is not None and self.status == "running":
            camera = self.capture_service.get_status()
            self.behavior = "Indeterminado"
            try:
                frame = self.capture_service.get_latest_jpeg()
                face_detections = self.face_recognition_service.recognize_jpeg(frame)
                behavior_tracks = self.pose_behavior_service.analyze_jpeg(frame, face_detections)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                face_detections = []
                behavior_tracks = []
                self.error = f"monitoring_ai_error: {exc}"

            self._latest_detections = behavior_tracks or face_detections
            best_detection = max(self._latest_detections, key=lambda item: item.confidence, default=None)
            display_detection = self._stable_display_detection(best_detection)
            if display_detection is not None:
                self._latest_detections = self._replace_matching_detection(self._latest_detections, display_detection)
            recognized_students_now = len(
                {
                    detection.student_name
                    for detection in self._latest_detections
                    if detection.recognized and detection.student_name != "Desconhecido"
                }
            )
            if display_detection is None:
                self.student_name = "Aguardando identificação"
                self.confidence = 0.0
                self.behavior = "Indeterminado"
            else:
                self.student_name = display_detection.student_name
                self.confidence = display_detection.confidence if display_detection.recognized else 0.0
                self.behavior = getattr(display_detection, "behavior", "Indeterminado")
            self.timestamp = datetime.now(UTC)
            await monitoring_event_hub.publish(
                "behavior_event",
                {
                    "session_id": self.session_id,
                    "student_name": self.student_name,
                    "behavior": self.behavior,
                    "confidence": self.confidence,
                    "camera_status": camera.status,
                    "websocket_status": "connected",
                    "faces_detected": len(face_detections),
                    "people_detected": len(behavior_tracks),
                    "recognized_students_now": recognized_students_now,
                    "bbox": display_detection.bbox if display_detection else None,
                    "timestamp": self.timestamp.isoformat(),
                },
            )
            await asyncio.sleep(1.0)

    def _stable_display_detection(
        self,
        candidate: FaceRecognitionResult | BehaviorTrack | None,
    ) -> FaceRecognitionResult | BehaviorTrack | None:
        now = time.monotonic()
        if candidate is None:
            if self._display_detection is not None and now - self._display_seen_at <= DETECTION_HOLD_SECONDS:
                return self._display_detection
            self._display_detection = None
            self._display_candidate_key = None
            self._display_candidate_count = 0
            return None

        propagated = self._propagate_current_identity(candidate, now)
        if propagated is not candidate:
            self._display_detection = propagated
            self._display_seen_at = now
            self._display_candidate_key = None
            self._display_candidate_count = 0
            return self._display_detection

        candidate_key = (candidate.student_name, getattr(candidate, "behavior", "Indeterminado"))
        current_key = (
            (self._display_detection.student_name, getattr(self._display_detection, "behavior", "Indeterminado"))
            if self._display_detection is not None
            else None
        )
        if current_key is None or candidate_key == current_key:
            self._display_detection = candidate
            self._display_seen_at = now
            self._display_candidate_key = None
            self._display_candidate_count = 0
            return self._display_detection

        if candidate_key == self._display_candidate_key:
            self._display_candidate_count += 1
        else:
            self._display_candidate_key = candidate_key
            self._display_candidate_count = 1

        if self._display_candidate_count >= BEHAVIOR_SWITCH_CONFIRMATIONS:
            self._display_detection = candidate
            self._display_seen_at = now
            self._display_candidate_key = None
            self._display_candidate_count = 0

        return self._display_detection

    def _propagate_current_identity(
        self,
        candidate: FaceRecognitionResult | BehaviorTrack,
        now: float,
    ) -> FaceRecognitionResult | BehaviorTrack:
        current = self._display_detection
        if current is None or not current.recognized:
            return candidate
        if candidate.recognized or candidate.student_name != "Desconhecido":
            return candidate
        if now - self._display_seen_at > IDENTITY_PROPAGATION_SECONDS:
            return candidate
        if not self._is_same_detection_area(current.bbox, candidate.bbox):
            return candidate

        confidence = max(0.0, min(float(current.confidence) * 0.96, float(current.confidence)))
        return replace(
            candidate,
            student_name=current.student_name,
            confidence=confidence,
            recognized=True,
        )

    def _replace_matching_detection(
        self,
        detections: list[FaceRecognitionResult | BehaviorTrack],
        display_detection: FaceRecognitionResult | BehaviorTrack,
    ) -> list[FaceRecognitionResult | BehaviorTrack]:
        for index, detection in enumerate(detections):
            if detection is display_detection or self._is_same_detection_area(detection.bbox, display_detection.bbox):
                updated = list(detections)
                updated[index] = display_detection
                return updated
        return [display_detection, *detections]

    def _is_same_detection_area(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        iou = self._box_iou(first, second)
        if iou >= IDENTITY_PROPAGATION_MIN_IOU:
            return True

        fcx, fcy, fdiag = self._box_center_and_diag(first)
        scx, scy, sdiag = self._box_center_and_diag(second)
        distance = ((fcx - scx) ** 2 + (fcy - scy) ** 2) ** 0.5
        scale = max(fdiag, sdiag, 1.0)
        return distance / scale <= IDENTITY_PROPAGATION_MAX_CENTER_DISTANCE

    def _box_iou(self, first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter_width = max(0, ix2 - ix1)
        inter_height = max(0, iy2 - iy1)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0
        first_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        second_area = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / max(1, first_area + second_area - intersection)

    def _box_center_and_diag(self, box: tuple[int, int, int, int]) -> tuple[float, float, float]:
        x1, y1, x2, y2 = box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return (x1 + width / 2.0, y1 + height / 2.0, (width * width + height * height) ** 0.5)

    def video_feed(self):
        while self.capture_service.get_status().status != "idle":
            frame = self.capture_service.get_latest_jpeg()
            if frame is None:
                time.sleep(0.1)
                continue
            annotated = self.face_recognition_service.annotate_jpeg(frame, self._latest_detections, self.behavior)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + (annotated or frame) + b"\r\n"
            time.sleep(0.04)


runtime_monitoring_service = MonitoringService()


def list_monitoring_options(db: Session, current_user: CurrentUser) -> MonitoringOptionsResponse:
    params: dict[str, object] = {}
    if current_user.role == "admin":
        subjects_query = """
            SELECT id, nome
            FROM subjects
            ORDER BY nome
        """
        classes_query = """
            SELECT id, nome, COALESCE(identificador, '') AS identificador
            FROM classes
            ORDER BY nome, identificador
        """
        assignments_query = """
            SELECT DISTINCT subject_id, class_id
            FROM teacher_subject_class
            ORDER BY subject_id, class_id
        """
    else:
        params["teacher_id"] = current_user.teacher_id
        subjects_query = """
            SELECT DISTINCT s.id, s.nome
            FROM subjects s
            JOIN teacher_subject_class tsc ON tsc.subject_id = s.id
            WHERE tsc.teacher_id = :teacher_id
            ORDER BY s.nome
        """
        classes_query = """
            SELECT DISTINCT c.id, c.nome, COALESCE(c.identificador, '') AS identificador
            FROM classes c
            JOIN teacher_subject_class tsc ON tsc.class_id = c.id
            WHERE tsc.teacher_id = :teacher_id
            ORDER BY c.nome, identificador
        """
        assignments_query = """
            SELECT DISTINCT subject_id, class_id
            FROM teacher_subject_class
            WHERE teacher_id = :teacher_id
            ORDER BY subject_id, class_id
        """

    subject_rows = db.execute(text(subjects_query), params).mappings().all()
    class_rows = db.execute(text(classes_query), params).mappings().all()
    assignment_rows = db.execute(text(assignments_query), params).mappings().all()
    return MonitoringOptionsResponse(
        subjects=[MonitoringSubject(**dict(row)) for row in subject_rows],
        classes=[MonitoringClass(**dict(row)) for row in class_rows],
        assignments=[dict(row) for row in assignment_rows],
        lesson_types=LESSON_TYPES,
    )


def ensure_teacher_can_monitor(db: Session, current_user: CurrentUser, subject_id: int, class_id: int) -> None:
    if current_user.role != "professor":
        return

    assignment = db.execute(
        text(
            """
            SELECT 1
            FROM teacher_subject_class
            WHERE teacher_id = :teacher_id
              AND subject_id = :subject_id
              AND class_id = :class_id
            LIMIT 1
            """
        ),
        {
            "teacher_id": current_user.teacher_id,
            "subject_id": subject_id,
            "class_id": class_id,
        },
    ).first()
    if assignment is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teacher assignment not found")


def get_or_create_monitoring_teacher_id(db: Session, current_user: CurrentUser) -> int:
    if current_user.teacher_id is not None:
        return current_user.teacher_id

    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User has no teacher profile")

    row = db.execute(
        text(
            """
            INSERT INTO teachers (user_id, nome)
            VALUES (:user_id, :nome)
            ON CONFLICT (user_id)
            DO UPDATE SET nome = EXCLUDED.nome
            RETURNING id
            """
        ),
        {"user_id": current_user.id, "nome": current_user.nome},
    ).mappings().one()
    return int(row["id"])


def create_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    payload: MonitoringSessionCreate,
) -> MonitoringSessionRead:
    ensure_teacher_can_monitor(db, current_user, payload.subject_id, payload.class_id)
    teacher_id = get_or_create_monitoring_teacher_id(db, current_user)

    row = db.execute(
        text(
            """
            INSERT INTO monitoring_sessions (
                teacher_id, subject_id, class_id, lesson_type, session_date, start_time, status
            )
            VALUES (
                :teacher_id, :subject_id, :class_id, :lesson_type, CURRENT_DATE, CURRENT_TIMESTAMP, 'em_andamento'
            )
            RETURNING id, status
            """
        ),
        {
            "teacher_id": teacher_id,
            "subject_id": payload.subject_id,
            "class_id": payload.class_id,
            "lesson_type": payload.lesson_type,
        },
    ).mappings().one()
    db.commit()
    return MonitoringSessionRead(id=row["id"], status=row["status"])


def close_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    session_id: int,
    close_status: str,
) -> MonitoringSessionRead:
    conditions = ["id = :session_id"]
    params: dict[str, object] = {"session_id": session_id, "status": close_status}
    if current_user.role == "professor":
        conditions.append("teacher_id = :teacher_id")
        params["teacher_id"] = current_user.teacher_id

    row = db.execute(
        text(
            f"""
            UPDATE monitoring_sessions
            SET end_time = CURRENT_TIMESTAMP, status = :status
            WHERE {" AND ".join(conditions)}
            RETURNING id, status
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitoring session not found")
    db.commit()
    return MonitoringSessionRead(id=row["id"], status=row["status"])


def get_monitoring_session(
    db: Session,
    current_user: CurrentUser,
    session_id: int,
) -> MonitoringSessionSummary:
    conditions = ["id = :session_id"]
    params: dict[str, object] = {"session_id": session_id}
    if current_user.role == "professor":
        conditions.append("teacher_id = :teacher_id")
        params["teacher_id"] = current_user.teacher_id

    row = db.execute(
        text(
            f"""
            SELECT id, status, start_time, end_time, lesson_type
            FROM monitoring_sessions
            WHERE {" AND ".join(conditions)}
            """
        ),
        params,
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Monitoring session not found")
    return MonitoringSessionSummary(**dict(row))
