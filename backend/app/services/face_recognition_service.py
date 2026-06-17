import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path


FACE_RECOGNITION_BASE_THRESHOLD = 0.46
FACE_RECOGNITION_MEDIUM_THRESHOLD = 0.41
FACE_RECOGNITION_SMALL_THRESHOLD = 0.34
FACE_RECOGNITION_TINY_THRESHOLD = 0.28
FACE_RECOGNITION_MIN_MARGIN = 0.035
FACE_RECOGNITION_TINY_MARGIN = 0.018
FACE_RECOGNITION_MIN_FACE_AREA = 1600
FACE_DETECTION_SIZE = int(os.getenv("FACE_DETECTION_SIZE", "960"))
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.35"))
SERVICE_PATH = Path(__file__).resolve()
PROJECT_ROOT = SERVICE_PATH.parents[2] if (SERVICE_PATH.parents[2] / "data").exists() else SERVICE_PATH.parents[3]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
NAMES_PATH = PROJECT_ROOT / "data" / "names.pkl"


@dataclass
class FaceRecognitionResult:
    student_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    recognized: bool


class FaceRecognitionService:
    """Small adapter that reuses the legacy InsightFace embeddings in FastAPI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._app = None
        self._cv2 = None
        self._np = None
        self._known_embeddings_norm = None
        self._known_names: list[str] = []
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def reload_embeddings(self) -> None:
        with self._lock:
            self._ensure_runtime_locked()
            self._load_embeddings_locked()

    def recognize_jpeg(self, frame_jpeg: bytes | None) -> list[FaceRecognitionResult]:
        if not frame_jpeg:
            return []

        with self._lock:
            self._ensure_runtime_locked()
            if self._cv2 is None or self._np is None or self._app is None:
                return []

            frame_array = self._np.frombuffer(frame_jpeg, dtype=self._np.uint8)
            frame = self._cv2.imdecode(frame_array, self._cv2.IMREAD_COLOR)
            if frame is None:
                return []

            rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            faces = self._app.get(rgb_frame)

            results: list[FaceRecognitionResult] = []
            for face in faces:
                student_name, confidence, recognized = self._identify_face_locked(face)
                x1, y1, x2, y2 = [int(value) for value in face.bbox]
                results.append(
                    FaceRecognitionResult(
                        student_name=student_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        recognized=recognized,
                    )
                )
            return results

    def annotate_jpeg(
        self,
        frame_jpeg: bytes | None,
        detections: list[FaceRecognitionResult],
        behavior: str,
    ) -> bytes | None:
        if not frame_jpeg:
            return None

        with self._lock:
            self._ensure_cv_runtime_locked()
            if self._cv2 is None or self._np is None:
                return frame_jpeg

            frame_array = self._np.frombuffer(frame_jpeg, dtype=self._np.uint8)
            frame = self._cv2.imdecode(frame_array, self._cv2.IMREAD_COLOR)
            if frame is None:
                return frame_jpeg

            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                current_behavior = getattr(detection, "behavior", behavior)
                is_negative = current_behavior in {"Agitado", "Dormindo", "Distraido"}
                color = (68, 68, 239) if is_negative else (85, 207, 98)
                label_bg = (36, 36, 170) if is_negative else (220, 252, 231)
                label_fg = (255, 255, 255) if is_negative else (45, 83, 20)
                label = f"{detection.student_name} - {current_behavior}"
                self._cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_size, baseline = self._cv2.getTextSize(label, self._cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                label_w, label_h = label_size
                pad_x = 10
                pad_y = 7
                label_x1 = max(0, x1)
                label_y2 = max(label_h + 2 * pad_y + 2, y1 - 8)
                label_x2 = min(frame.shape[1] - 1, label_x1 + label_w + 2 * pad_x)
                label_y1 = max(0, label_y2 - label_h - 2 * pad_y)
                self._draw_rounded_rect(frame, (label_x1, label_y1), (label_x2, label_y2), label_bg, radius=8)
                self._cv2.putText(
                    frame,
                    label,
                    (label_x1 + pad_x, label_y2 - pad_y - baseline + 2),
                    self._cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    label_fg,
                    2,
                    self._cv2.LINE_AA,
                )

            encoded, jpg = self._cv2.imencode(".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encoded:
                return frame_jpeg
            return jpg.tobytes()

    def _draw_rounded_rect(self, frame, top_left, bottom_right, color, radius: int = 8) -> None:
        x1, y1 = top_left
        x2, y2 = bottom_right
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if radius <= 0:
            self._cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            return
        self._cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        self._cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        self._cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        self._cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        self._cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        self._cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)

    def _ensure_runtime_locked(self) -> None:
        self._ensure_cv_runtime_locked()
        if self._app is not None:
            return

        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(FACE_DETECTION_SIZE, FACE_DETECTION_SIZE), det_thresh=FACE_DETECTION_THRESHOLD)
            self._load_embeddings_locked()
        except Exception as exc:  # pragma: no cover - depends on model runtime
            self._load_error = f"insightface_unavailable: {exc}"
            self._app = None

    def _ensure_cv_runtime_locked(self) -> None:
        if self._cv2 is not None and self._np is not None:
            return
        try:
            import cv2
            import numpy as np

            self._cv2 = cv2
            self._np = np
        except Exception as exc:  # pragma: no cover - optional runtime deps
            self._load_error = f"opencv_or_numpy_unavailable: {exc}"
            self._cv2 = None
            self._np = None

    def _load_embeddings_locked(self) -> None:
        if self._np is None:
            return

        try:
            if not EMBEDDINGS_PATH.exists() or not NAMES_PATH.exists():
                raise FileNotFoundError(f"{EMBEDDINGS_PATH} or {NAMES_PATH}")

            embeddings = self._np.load(EMBEDDINGS_PATH)
            with NAMES_PATH.open("rb") as names_file:
                names = pickle.load(names_file)
            if len(embeddings) == 0:
                self._known_embeddings_norm = None
                self._known_names = []
                return

            embeddings = self._np.asarray(embeddings, dtype=self._np.float32)
            self._known_embeddings_norm = embeddings / (self._np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
            self._known_names = [str(name) for name in names]
            self._load_error = None
        except Exception as exc:
            self._known_embeddings_norm = None
            self._known_names = []
            self._load_error = f"embeddings_unavailable: {exc}"

    def _identify_face_locked(self, face) -> tuple[str, float, bool]:
        if self._known_embeddings_norm is None or not self._known_names or self._np is None:
            return "Desconhecido", 0.0, False

        emb = face.embedding
        emb = emb / (self._np.linalg.norm(emb) + 1e-6)
        sims = self._np.dot(self._known_embeddings_norm, emb)
        best_idx = int(self._np.argmax(sims))
        best_score = float(sims[best_idx])

        second_best_score = -1.0
        if len(sims) > 1:
            second_best_score = float(self._np.partition(sims, -2)[-2])

        x1, y1, x2, y2 = face.bbox.astype(int)
        face_area = max(1, (x2 - x1) * (y2 - y1))
        if face_area < FACE_RECOGNITION_MIN_FACE_AREA:
            return "Desconhecido", best_score, False

        if face_area < 3500:
            acceptance_threshold = FACE_RECOGNITION_TINY_THRESHOLD
            min_margin = FACE_RECOGNITION_TINY_MARGIN
        elif face_area < 7000:
            acceptance_threshold = FACE_RECOGNITION_SMALL_THRESHOLD
            min_margin = FACE_RECOGNITION_MIN_MARGIN * 0.7
        elif face_area < 14000:
            acceptance_threshold = FACE_RECOGNITION_MEDIUM_THRESHOLD
            min_margin = FACE_RECOGNITION_MIN_MARGIN
        else:
            acceptance_threshold = FACE_RECOGNITION_BASE_THRESHOLD
            min_margin = FACE_RECOGNITION_MIN_MARGIN

        margin = best_score - second_best_score if second_best_score >= 0 else best_score
        if best_score >= acceptance_threshold and margin >= min_margin:
            return self._known_names[best_idx], best_score, True

        return "Desconhecido", best_score, False
