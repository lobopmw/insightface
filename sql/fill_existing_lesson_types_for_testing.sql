BEGIN;

UPDATE monitoring_sessions
SET lesson_type = CASE
    WHEN MOD(id, 2) = 0 THEN 'Atividade'
    ELSE 'Prova'
END
WHERE lesson_type IS NULL OR BTRIM(lesson_type) = '';

UPDATE behavior_episode be
SET lesson_type = ms.lesson_type
FROM monitoring_sessions ms
WHERE be.monitoring_session_id = ms.id
  AND (be.lesson_type IS NULL OR BTRIM(be.lesson_type) = '');

COMMIT;
