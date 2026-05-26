from datetime import datetime, timedelta

from app.ai.behavior import BehaviorEpisodeManager


def test_behavior_episode_manager_persists_after_stability_frames() -> None:
    persisted: list[dict] = []
    manager = BehaviorEpisodeManager(
        persist_callback=lambda **kwargs: persisted.append(kwargs),
        stability_seconds=999,
        stability_frames=2,
    )
    start = datetime(2026, 1, 1, 8, 0, 0)

    manager.update_behavior("s1", "Aluno 1", "s1", "Atento", start, "Escola", "Matematica", "Professor")
    manager.update_behavior(
        "s1",
        "Aluno 1",
        "s1",
        "Distraido",
        start + timedelta(seconds=1),
        "Escola",
        "Matematica",
        "Professor",
    )
    manager.update_behavior(
        "s1",
        "Aluno 1",
        "s1",
        "Distraido",
        start + timedelta(seconds=2),
        "Escola",
        "Matematica",
        "Professor",
    )

    assert len(persisted) == 1
    assert persisted[0]["behavior"] == "Atento"
