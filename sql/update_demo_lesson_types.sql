BEGIN;

UPDATE monitoring_sessions
SET lesson_type = CASE id
    WHEN 9001 THEN 'Atividade'
    WHEN 9002 THEN 'Prova'
    WHEN 9003 THEN 'Atividade'
    WHEN 9004 THEN 'Prova'
    WHEN 9005 THEN 'Atividade'
    WHEN 9006 THEN 'Prova'
    ELSE lesson_type
END
WHERE id IN (9001, 9002, 9003, 9004, 9005, 9006);

UPDATE behavior_episode be
SET lesson_type = ms.lesson_type
FROM monitoring_sessions ms
WHERE be.monitoring_session_id = ms.id
  AND ms.id IN (9001, 9002, 9003, 9004, 9005, 9006);

COMMIT;
