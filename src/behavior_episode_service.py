from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional


@dataclass
class StudentEpisodeState:
    student_name: str
    student_id: Optional[str]
    current_behavior: str
    current_start_time: datetime
    pending_behavior: Optional[str] = None
    pending_since: Optional[datetime] = None
    pending_frames: int = 0


class BehaviorEpisodeManager:
    """Gerencia episodios por aluno com debounce por tempo ou frames."""

    def __init__(
        self,
        persist_callback: Callable[..., None],
        stability_seconds: float = 2.0,
        stability_frames: int = 15,
    ):
        self.persist_callback = persist_callback
        self.stability_seconds = float(stability_seconds)
        self.stability_frames = int(stability_frames)
        self.states: Dict[str, StudentEpisodeState] = {}
        self.last_persist_error: Optional[str] = None

    @staticmethod
    def _should_persist_behavior(behavior: str | None) -> bool:
        normalized = str(behavior or "").strip().lower()
        return normalized not in {"", "indeterminado"}

    def update_behavior(
        self,
        student_key: str,
        student_name: str,
        student_id: Optional[str],
        behavior: str,
        timestamp: datetime,
        school: str,
        discipline: str,
        teacher: str,
        source: str = "realtime",
    ):
        state = self.states.get(student_key)

        if state is None:
            self.states[student_key] = StudentEpisodeState(
                student_name=student_name,
                student_id=student_id,
                current_behavior=behavior,
                current_start_time=timestamp,
            )
            return

        state.student_name = student_name
        state.student_id = student_id

        if behavior == state.current_behavior:
            state.pending_behavior = None
            state.pending_since = None
            state.pending_frames = 0
            return

        if state.pending_behavior != behavior:
            state.pending_behavior = behavior
            state.pending_since = timestamp
            state.pending_frames = 1
            return

        state.pending_frames += 1
        elapsed = (timestamp - (state.pending_since or timestamp)).total_seconds()

        if elapsed >= self.stability_seconds or state.pending_frames >= self.stability_frames:
            self._close_current_episode(
                student_name=student_name,
                student_id=student_id,
                school=school,
                discipline=discipline,
                teacher=teacher,
                behavior=state.current_behavior,
                start_time=state.current_start_time,
                end_time=timestamp,
                source=source,
            )

            state.current_behavior = behavior
            state.current_start_time = timestamp
            state.pending_behavior = None
            state.pending_since = None
            state.pending_frames = 0

    def flush_student(
        self,
        student_key: str,
        student_name: str,
        student_id: Optional[str],
        timestamp: datetime,
        school: str,
        discipline: str,
        teacher: str,
        source: str = "realtime",
    ):
        state = self.states.get(student_key)
        if not state:
            return

        self._close_current_episode(
            student_name=student_name,
            student_id=student_id,
            school=school,
            discipline=discipline,
            teacher=teacher,
            behavior=state.current_behavior,
            start_time=state.current_start_time,
            end_time=timestamp,
            source=source,
        )

    def flush_all(
        self,
        timestamp: datetime,
        school: str,
        discipline: str,
        teacher: str,
        source: str = "realtime",
    ):
        for student_key, state in list(self.states.items()):
            self._close_current_episode(
                student_name=state.student_name or student_key,
                student_id=state.student_id,
                school=school,
                discipline=discipline,
                teacher=teacher,
                behavior=state.current_behavior,
                start_time=state.current_start_time,
                end_time=timestamp,
                source=source,
            )

    def _close_current_episode(
        self,
        student_name: str,
        student_id: Optional[str],
        school: str,
        discipline: str,
        teacher: str,
        behavior: str,
        start_time: datetime,
        end_time: datetime,
        source: str,
    ):
        if not self._should_persist_behavior(behavior):
            return
        try:
            self.persist_callback(
                school=school,
                discipline=discipline,
                teacher=teacher,
                id_student=student_id,
                student=student_name,
                behavior=behavior,
                start_time=start_time,
                end_time=end_time,
                source=source,
            )
            self.last_persist_error = None
        except Exception as exc:
            self.last_persist_error = str(exc)
            print(f"[episode_manager] persist error: {exc}")
