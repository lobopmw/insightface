import os
import time
import threading
from dataclasses import dataclass

from app.services.face_recognition_service import FaceRecognitionResult


ENTER_SLEEP_FRAMES = 6
EXIT_SLEEP_FRAMES = 10
ENTER_QUESTION_FRAMES = 4
ENTER_ATTENTIVE_FRAMES = 5
ENTER_AGITATED_FRAMES = 3
ENTER_DISTRACTED_FRAMES = int(os.getenv("ENTER_DISTRACTED_FRAMES", "3"))
ENTER_UNDETERMINED_FRAMES = 2
DISTRACTED_TIMEOUT_SECONDS = float(os.getenv("DISTRACTED_TIMEOUT_SECONDS", "2.5"))
DISPLAY_BOX_MIN_KEYPOINT_CONF = 0.22
DISPLAY_BOX_MIN_WIDTH = 56.0


@dataclass
class BehaviorTrack:
    student_name: str
    behavior: str
    confidence: float
    bbox: tuple[int, int, int, int]
    recognized: bool


class PoseBehaviorService:
    """YOLOv11-Pose adapter using the same geometric rules from the Streamlit runtime."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("POSE_MODEL_NAME", "yolo11n-pose.pt")
        self.pose_imgsz = int(os.getenv("POSE_IMGSZ_CPU", "800"))
        self.pose_conf = float(os.getenv("POSE_DET_CONF", "0.18"))
        self._lock = threading.Lock()
        self._cv2 = None
        self._np = None
        self._model = None
        self._load_error: str | None = None
        self._lateral_timers: dict[str, dict[str, object]] = {}
        self._state_smoother: dict[str, dict[str, object]] = {}

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def analyze_jpeg(
        self,
        frame_jpeg: bytes | None,
        faces: list[FaceRecognitionResult],
    ) -> list[BehaviorTrack]:
        if not frame_jpeg:
            return []

        with self._lock:
            self._ensure_runtime_locked()
            if self._cv2 is None or self._np is None or self._model is None:
                return []

            frame_array = self._np.frombuffer(frame_jpeg, dtype=self._np.uint8)
            frame = self._cv2.imdecode(frame_array, self._cv2.IMREAD_COLOR)
            if frame is None:
                return []

            results = self._model.predict(
                frame,
                show=False,
                device="cpu",
                verbose=False,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
            )
            return self._tracks_from_results(results, faces, frame.shape[:2])

    def _ensure_runtime_locked(self) -> None:
        if self._cv2 is None or self._np is None:
            try:
                import cv2
                import numpy as np

                self._cv2 = cv2
                self._np = np
            except Exception as exc:  # pragma: no cover - optional deps
                self._load_error = f"opencv_or_numpy_unavailable: {exc}"
                return

        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            self._load_error = None
        except Exception as exc:  # pragma: no cover - depends on installed runtime/model
            self._model = None
            self._load_error = f"pose_model_unavailable: {exc}"

    def _tracks_from_results(
        self,
        results,
        faces: list[FaceRecognitionResult],
        frame_shape: tuple[int, int],
    ) -> list[BehaviorTrack]:
        tracks: list[BehaviorTrack] = []
        for result in results or []:
            if not hasattr(result, "keypoints") or result.keypoints is None or len(result.keypoints) == 0:
                continue
            keypoints_all = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy() if getattr(result, "boxes", None) is not None else []

            for pid, person_keypoints in enumerate(keypoints_all):
                if len(person_keypoints) == 0:
                    continue

                pose_conf_threshold = self._adaptive_keypoint_conf_threshold(person_keypoints)
                person_box = self._person_box_from_keypoints(person_keypoints, pose_conf_threshold)
                detector_box = self._clamp_box_to_frame(boxes[pid], frame_shape) if pid < len(boxes) else None
                fallback_box = detector_box or person_box
                if fallback_box is None:
                    continue
                if self._is_back_facing_person(person_keypoints, pose_conf_threshold):
                    continue
                if not self._has_front_or_side_face(person_keypoints, pose_conf_threshold):
                    continue

                face = self._match_face_to_person(fallback_box, faces)
                if face is None:
                    continue
                student_name = face.student_name if face is not None else "Desconhecido"
                confidence = face.confidence if face is not None and face.recognized else 0.0
                recognized = bool(face is not None and face.recognized)
                behavior_key = student_name if recognized else f"pid_{pid}"
                behavior = self._classify_person_behavior(person_keypoints, pose_conf_threshold, behavior_key)
                display_box = self._build_display_box(
                    person_keypoints,
                    behavior,
                    pose_conf_threshold,
                    fallback_box=fallback_box,
                    frame_shape=frame_shape,
                )
                if display_box is None:
                    continue
                tracks.append(
                    BehaviorTrack(
                        student_name=student_name,
                        behavior=behavior,
                        confidence=confidence,
                        bbox=display_box,
                        recognized=recognized,
                    )
                )
        return tracks

    def _classify_person_behavior(self, person_keypoints, threshold: float, behavior_key: str) -> str:
        current_behavior = "Indeterminado"
        have_all = False

        if person_keypoints.shape[0] > 10:
            nose = person_keypoints[0]
            ls, rs = person_keypoints[5], person_keypoints[6]
            le, re = person_keypoints[7], person_keypoints[8]
            lw, rw = person_keypoints[9], person_keypoints[10]
            confs = [p[2] for p in [nose, ls, rs, le, re, lw, rw]]
            have_all = all(c > threshold for c in confs)
            if have_all:
                current_behavior = self._classify_behavior(nose, ls, rs, le, re, lw, rw, threshold)

            if person_keypoints.shape[0] > 6:
                l_eye = person_keypoints[1]
                r_eye = person_keypoints[2]
                l_ear = person_keypoints[3]
                r_ear = person_keypoints[4]
                if nose[2] > threshold or (ls[2] > threshold and rs[2] > threshold):
                    lateral_status = self._is_lateral_view(nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=threshold)
                    sleep_like_posture = False
                    if have_all:
                        sleep_metrics = self._analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, threshold)
                        sleep_like_posture = sleep_metrics["strong_sleep"] or sleep_metrics["head_supported"]
                    distracted = self._check_distracted_status(
                        behavior_key,
                        lateral_status and not sleep_like_posture,
                        timeout=DISTRACTED_TIMEOUT_SECONDS,
                    )
                    if distracted:
                        current_behavior = distracted

        return self._smooth_behavior(behavior_key, current_behavior)

    def _smooth_behavior(self, key: str, raw_behavior: str) -> str:
        state = self._state_smoother.setdefault(
            key,
            {"state": "Indeterminado", "candidate": None, "candidate_count": 0},
        )
        if raw_behavior == state["state"]:
            state["candidate"] = None
            state["candidate_count"] = 0
        else:
            if raw_behavior == state["candidate"]:
                state["candidate_count"] = int(state["candidate_count"]) + 1
            else:
                state["candidate"] = raw_behavior
                state["candidate_count"] = 1

            required_frames = self._transition_frames_required(str(state["state"]), raw_behavior)
            if int(state["candidate_count"]) >= required_frames:
                state["state"] = raw_behavior
                state["candidate"] = None
                state["candidate_count"] = 0
        return str(state["state"])

    def _person_box_from_keypoints(self, person_keypoints, threshold: float):
        visible = [p for p in person_keypoints if len(p) >= 3 and float(p[2]) > threshold]
        if not visible:
            return None
        xs = [float(p[0]) for p in visible]
        ys = [float(p[1]) for p in visible]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def _match_face_to_person(self, person_box, faces: list[FaceRecognitionResult]) -> FaceRecognitionResult | None:
        best_iou = 0.0
        best_face = None
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            face_center_x = (fx1 + fx2) / 2.0
            face_center_y = (fy1 + fy2) / 2.0
            px1, py1, px2, py2 = person_box
            if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                return face
            score = self._iou(person_box, face.bbox)
            if score > best_iou:
                best_iou = score
                best_face = face
        return best_face if best_iou >= 0.03 else None

    def _adaptive_keypoint_conf_threshold(self, person_keypoints) -> float:
        visible_points = [p for p in person_keypoints if len(p) >= 3 and float(p[2]) > 0.08]
        if not visible_points:
            return 0.22
        xs = [float(p[0]) for p in visible_points]
        ys = [float(p[1]) for p in visible_points]
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        if size < 110:
            return 0.14
        if size < 180:
            return 0.17
        if size < 260:
            return 0.19
        return 0.22

    def _visible_keypoint_xy(self, person_keypoints, idx: int, threshold: float):
        if idx < 0 or idx >= len(person_keypoints):
            return None
        point = person_keypoints[idx]
        if len(point) < 3 or float(point[2]) <= float(threshold):
            return None
        return (float(point[0]), float(point[1]))

    def _build_display_box(self, person_keypoints, behavior: str, threshold: float, fallback_box=None, frame_shape=None):
        geometry_threshold = max(DISPLAY_BOX_MIN_KEYPOINT_CONF, float(threshold))
        nose = self._visible_keypoint_xy(person_keypoints, 0, geometry_threshold)
        ls = self._visible_keypoint_xy(person_keypoints, 5, geometry_threshold)
        rs = self._visible_keypoint_xy(person_keypoints, 6, geometry_threshold)
        lh = self._visible_keypoint_xy(person_keypoints, 11, geometry_threshold)
        rh = self._visible_keypoint_xy(person_keypoints, 12, geometry_threshold)

        torso_points = [point for point in (nose, ls, rs, lh, rh) if point is not None]
        if len(torso_points) < 2:
            return self._clamp_box_to_frame(fallback_box, frame_shape) if fallback_box is not None else None

        if ls is not None and rs is not None:
            center_x = (ls[0] + rs[0]) / 2.0
            shoulder_y = (ls[1] + rs[1]) / 2.0
            shoulder_span = max(DISPLAY_BOX_MIN_WIDTH, abs(rs[0] - ls[0]))
            torso_x_min = min(ls[0], rs[0])
            torso_x_max = max(ls[0], rs[0])
        else:
            xs = [point[0] for point in torso_points]
            ys = [point[1] for point in torso_points]
            center_x = sum(xs) / len(xs)
            shoulder_y = min(ys) + 0.35 * DISPLAY_BOX_MIN_WIDTH
            shoulder_span = max(DISPLAY_BOX_MIN_WIDTH, max(xs) - min(xs))
            torso_x_min = min(xs)
            torso_x_max = max(xs)

        hip_points = [point for point in (lh, rh) if point is not None]
        if hip_points:
            torso_x_min = min([torso_x_min] + [point[0] for point in hip_points])
            torso_x_max = max([torso_x_max] + [point[0] for point in hip_points])
            bottom = max(point[1] for point in hip_points) + 0.28 * shoulder_span
        else:
            bottom = shoulder_y + 2.15 * shoulder_span

        top = nose[1] - 0.55 * shoulder_span if nose is not None else shoulder_y - 0.95 * shoulder_span
        half_width = max(0.82 * shoulder_span, 0.72 * max(1.0, torso_x_max - torso_x_min), DISPLAY_BOX_MIN_WIDTH / 2.0)
        left = center_x - half_width
        right = center_x + half_width

        if behavior in ("Perguntando", "Agitado"):
            shoulder_ceiling = min(point[1] for point in (ls, rs) if point is not None) if (ls or rs) else shoulder_y
            arm_points = []
            for idx in (7, 8, 9, 10):
                point = self._visible_keypoint_xy(person_keypoints, idx, geometry_threshold)
                if point is not None and point[1] < shoulder_ceiling + 0.55 * shoulder_span:
                    arm_points.append(point)
            if arm_points:
                left = min(left, min(point[0] for point in arm_points) - 0.18 * shoulder_span)
                right = max(right, max(point[0] for point in arm_points) + 0.18 * shoulder_span)
                top = min(top, min(point[1] for point in arm_points) - 0.18 * shoulder_span)

        return self._clamp_box_to_frame((left, top, right, bottom), frame_shape)

    def _classify_behavior(self, nose, ls, rs, le, re, lw, rw, threshold) -> str:
        cx = (ls[0] + rs[0]) / 2.0
        cy = (ls[1] + rs[1]) / 2.0
        s = max(1.0, float(abs(ls[0] - rs[0])))
        head_clearance = cy - nose[1]
        head_offset_x = abs(nose[0] - cx)
        shoulder_tilt = abs(ls[1] - rs[1])
        sleep_metrics = self._analyze_sleep_posture(nose, ls, rs, le, re, lw, rw, threshold)

        is_head_low = head_clearance < 0.16 * s or nose[1] > cy - 0.03 * s
        is_looking_down = head_clearance < 0.10 * s or nose[1] > cy + 0.06 * s
        left_hand_near_head = lw[2] > threshold and (
            self._point_distance(lw, nose) < 0.34 * s
            or (abs(lw[0] - nose[0]) < 0.24 * s and abs(lw[1] - nose[1]) < 0.30 * s)
        )
        right_hand_near_head = rw[2] > threshold and (
            self._point_distance(rw, nose) < 0.34 * s
            or (abs(rw[0] - nose[0]) < 0.24 * s and abs(rw[1] - nose[1]) < 0.30 * s)
        )

        if s >= 28.0 and (sleep_metrics["strong_sleep"] or sleep_metrics["moderate_sleep"]):
            return "Dormindo"
        if is_looking_down or is_head_low:
            return "Indeterminado"

        raised_margin = 0.22 * s
        min_head_gap_x = 0.28 * s
        elbow_margin = 0.04 * s
        up_l = (
            lw[2] > threshold
            and le[2] > threshold
            and lw[1] < cy - raised_margin
            and lw[1] < le[1] - elbow_margin
            and abs(lw[0] - nose[0]) > min_head_gap_x
            and not left_hand_near_head
        )
        up_r = (
            rw[2] > threshold
            and re[2] > threshold
            and rw[1] < cy - raised_margin
            and rw[1] < re[1] - elbow_margin
            and abs(rw[0] - nose[0]) > min_head_gap_x
            and not right_hand_near_head
        )
        if up_l and up_r:
            return "Agitado" if abs(lw[0] - rw[0]) > 0.95 * s else "Perguntando"
        if up_l or up_r:
            return "Perguntando"

        hands_compact_front = (
            lw[2] > threshold
            and rw[2] > threshold
            and abs(lw[0] - rw[0]) < 0.45 * s
            and min(lw[1], rw[1]) > nose[1]
            and max(lw[1], rw[1]) < cy + 0.45 * s
        )
        phone_like_posture = hands_compact_front and head_clearance < 0.28 * s
        attentive_posture = (
            head_clearance > 0.24 * s
            and head_offset_x < 0.34 * s
            and shoulder_tilt < max(16.0, 0.18 * s)
            and not phone_like_posture
            and not left_hand_near_head
            and not right_hand_near_head
        )
        if attentive_posture:
            return "Atento"
        return "Indeterminado"

    def _is_lateral_view(self, nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=0.5, cam_side="LEFT", cam_offset=0.12) -> bool:
        s = float(abs(ls[0] - rs[0])) + 1e-6
        t_ratio = 0.18 if s < 40 else 0.24 if s < 60 else 0.30
        offset = -cam_offset if cam_side == "LEFT" else cam_offset if cam_side == "RIGHT" else 0.0
        eye_dx = float(abs(l_eye[0] - r_eye[0]))
        cond_ratio_eyes = False
        if eye_dx >= 1.0:
            ratio_eyes = abs(nose[0] - (l_eye[0] + r_eye[0]) / 2.0) / eye_dx
            cond_ratio_eyes = abs(ratio_eyes + offset) > t_ratio
        cond_conf_eyes = (
            (l_eye[2] < conf_thr and r_eye[2] > conf_thr + 0.08)
            or (r_eye[2] < conf_thr and l_eye[2] > conf_thr + 0.08)
        )
        cond_ears = False
        if (l_ear[2] > conf_thr) or (r_ear[2] > conf_thr):
            ear_dx = float(abs(l_ear[0] - r_ear[0]))
            cond_ratio_ears = False
            if ear_dx >= 1.0:
                ratio_ears = abs(nose[0] - (l_ear[0] + r_ear[0]) / 2.0) / ear_dx
                cond_ratio_ears = ratio_ears + offset > (t_ratio * 0.90)
            cond_conf_ears = (
                (l_ear[2] < conf_thr and r_ear[2] > conf_thr + 0.08)
                or (r_ear[2] < conf_thr and l_ear[2] > conf_thr + 0.08)
            )
            cond_ears = cond_ratio_ears or cond_conf_ears
        return bool(cond_ratio_eyes or cond_conf_eyes or cond_ears)

    def _is_back_view(self, nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=0.5) -> bool:
        shoulder_width = float(abs(ls[0] - rs[0]))
        if ls[2] <= conf_thr or rs[2] <= conf_thr or shoulder_width < 35.0:
            return False
        frontal_face_missing = nose[2] < conf_thr and l_eye[2] < conf_thr and r_eye[2] < conf_thr
        ears_missing = l_ear[2] < conf_thr and r_ear[2] < conf_thr
        shoulder_balance = abs(ls[1] - rs[1]) < max(18.0, 0.35 * shoulder_width)
        if frontal_face_missing and ears_missing and shoulder_balance:
            return True

        face_conf = max(float(nose[2]), float(l_eye[2]), float(r_eye[2]))
        ear_conf = max(float(l_ear[2]), float(r_ear[2]))
        shoulder_center_x = (float(ls[0]) + float(rs[0])) / 2.0
        head_x_candidates = [float(point[0]) for point in (nose, l_eye, r_eye, l_ear, r_ear) if float(point[2]) > 0.08]
        head_center_x = sum(head_x_candidates) / len(head_x_candidates) if head_x_candidates else shoulder_center_x
        head_centered = abs(head_center_x - shoulder_center_x) / max(1.0, shoulder_width) < 0.24
        weak_frontal_face = face_conf < conf_thr + 0.18
        visible_back_head = ear_conf > conf_thr + 0.35
        return bool(shoulder_balance and head_centered and weak_frontal_face and visible_back_head)

    def _is_back_facing_person(self, person_keypoints, threshold: float) -> bool:
        if person_keypoints.shape[0] <= 6:
            return False
        nose = person_keypoints[0]
        l_eye = person_keypoints[1]
        r_eye = person_keypoints[2]
        l_ear = person_keypoints[3]
        r_ear = person_keypoints[4]
        ls = person_keypoints[5]
        rs = person_keypoints[6]
        return self._is_back_view(nose, l_eye, r_eye, l_ear, r_ear, ls, rs, conf_thr=threshold)

    def _has_front_or_side_face(self, person_keypoints, threshold: float) -> bool:
        if person_keypoints.shape[0] <= 6:
            return False

        nose = person_keypoints[0]
        l_eye = person_keypoints[1]
        r_eye = person_keypoints[2]
        l_ear = person_keypoints[3]
        r_ear = person_keypoints[4]
        ls = person_keypoints[5]
        rs = person_keypoints[6]

        shoulder_width = float(abs(ls[0] - rs[0]))
        if shoulder_width < 35.0:
            return False

        front_face = (
            float(nose[2]) > threshold + 0.08
            and float(l_eye[2]) > threshold
            and float(r_eye[2]) > threshold
        )
        side_face = (
            float(nose[2]) > threshold + 0.16
            and (
                max(float(l_eye[2]), float(r_eye[2])) > threshold + 0.16
                or max(float(l_ear[2]), float(r_ear[2])) > threshold + 0.48
            )
        )
        return bool(front_face or side_face)

    def _check_distracted_status(self, name: str, is_distracted_pose: bool, timeout: float) -> str | None:
        now = time.time()
        state = self._lateral_timers.setdefault(name, {"start_time": None, "is_lateral": False})
        if is_distracted_pose:
            if not state["is_lateral"] and state["start_time"] is None:
                state["start_time"] = now
                state["is_lateral"] = True
            else:
                elapsed = now - float(state["start_time"] or now)
                if elapsed >= timeout:
                    return "Distraido"
        else:
            state["start_time"] = None
            state["is_lateral"] = False
        return None

    def _analyze_sleep_posture(self, nose, ls, rs, le, re, lw, rw, threshold):
        cx = (ls[0] + rs[0]) / 2.0
        cy = (ls[1] + rs[1]) / 2.0
        s = max(1.0, float(abs(ls[0] - rs[0])))
        left_wrist_visible = lw[2] > threshold
        right_wrist_visible = rw[2] > threshold
        left_elbow_visible = le[2] > threshold
        right_elbow_visible = re[2] > threshold
        nose_below_shoulders = nose[1] > cy + 0.02 * s
        nose_far_below_shoulders = nose[1] > cy + 0.12 * s
        nose_centered = abs(nose[0] - cx) < 0.40 * s
        left_wrist_near_nose = left_wrist_visible and (
            self._point_distance(lw, nose) < 0.38 * s
            or (abs(lw[0] - nose[0]) < 0.28 * s and abs(lw[1] - nose[1]) < 0.32 * s)
        )
        right_wrist_near_nose = right_wrist_visible and (
            self._point_distance(rw, nose) < 0.38 * s
            or (abs(rw[0] - nose[0]) < 0.28 * s and abs(rw[1] - nose[1]) < 0.32 * s)
        )
        left_elbow_near_nose = left_elbow_visible and (
            self._point_distance(le, nose) < 0.44 * s
            or (abs(le[0] - nose[0]) < 0.34 * s and abs(le[1] - nose[1]) < 0.26 * s)
        )
        right_elbow_near_nose = right_elbow_visible and (
            self._point_distance(re, nose) < 0.44 * s
            or (abs(re[0] - nose[0]) < 0.34 * s and abs(re[1] - nose[1]) < 0.26 * s)
        )
        left_arm_support = left_wrist_visible and left_elbow_visible and abs(lw[0] - le[0]) < 0.42 * s and abs(lw[1] - le[1]) < 0.34 * s
        right_arm_support = right_wrist_visible and right_elbow_visible and abs(rw[0] - re[0]) < 0.42 * s and abs(rw[1] - re[1]) < 0.34 * s
        wrists_below_shoulders = sum(
            1 for wrist, visible in ((lw, left_wrist_visible), (rw, right_wrist_visible)) if visible and wrist[1] > cy - 0.02 * s
        )
        elbows_below_shoulders = sum(
            1 for elbow, visible in ((le, left_elbow_visible), (re, right_elbow_visible)) if visible and elbow[1] > cy - 0.08 * s
        )
        wrist_near_count = int(left_wrist_near_nose) + int(right_wrist_near_nose)
        elbow_near_count = int(left_elbow_near_nose) + int(right_elbow_near_nose)
        support_count = int(left_arm_support) + int(right_arm_support)
        head_supported = (
            (left_wrist_near_nose and (left_arm_support or left_elbow_near_nose))
            or (right_wrist_near_nose and (right_arm_support or right_elbow_near_nose))
            or (wrist_near_count >= 1 and elbow_near_count >= 1)
        )
        moderate_sleep = (
            nose_below_shoulders
            and wrists_below_shoulders >= 1
            and elbows_below_shoulders >= 1
            and (
                head_supported
                or (wrist_near_count >= 1 and nose_centered)
                or (elbow_near_count >= 1 and nose_far_below_shoulders)
                or (support_count >= 1 and nose_far_below_shoulders)
            )
        )
        strong_sleep = (
            nose_below_shoulders
            and nose_centered
            and wrists_below_shoulders >= 1
            and elbows_below_shoulders >= 1
            and (
                (wrist_near_count >= 1 and elbow_near_count >= 1)
                or (wrist_near_count >= 1 and support_count >= 1)
                or (nose_far_below_shoulders and elbow_near_count >= 1)
            )
        )
        return {
            "scale": s,
            "head_supported": head_supported,
            "moderate_sleep": moderate_sleep,
            "strong_sleep": strong_sleep,
        }

    def _transition_frames_required(self, current_state: str, candidate_state: str) -> int:
        if current_state == "Dormindo" and candidate_state != "Dormindo":
            return EXIT_SLEEP_FRAMES
        return {
            "Dormindo": ENTER_SLEEP_FRAMES,
            "Perguntando": ENTER_QUESTION_FRAMES,
            "Atento": ENTER_ATTENTIVE_FRAMES,
            "Agitado": ENTER_AGITATED_FRAMES,
            "Distraido": ENTER_DISTRACTED_FRAMES,
            "Indeterminado": ENTER_UNDETERMINED_FRAMES,
        }.get(candidate_state, ENTER_UNDETERMINED_FRAMES)

    def _point_distance(self, p1, p2) -> float:
        return float(self._np.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))

    def _clamp_box_to_frame(self, box, frame_shape=None):
        if box is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in box]
        if frame_shape is not None:
            h, w = frame_shape
            x1 = max(0.0, min(x1, w - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))
        if x2 <= x1 or y2 <= y1:
            return None
        return int(x1), int(y1), int(x2), int(y2)

    def _iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, area_a + area_b - inter)
