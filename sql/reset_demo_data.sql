BEGIN;

DELETE FROM behavior_episode
WHERE source = 'seed_demo';

DELETE FROM monitoring_sessions
WHERE id IN (9001, 9002, 9003, 9004, 9005, 9006);

DELETE FROM teacher_subject_class
WHERE teacher_id IN (8001, 8002)
   OR subject_id IN (7001, 7002)
   OR class_id IN (6001, 6002);

UPDATE students
SET class_id = NULL
WHERE id IN ('demo_aluno_01', 'demo_aluno_02', 'demo_aluno_03', 'demo_aluno_04', 'demo_aluno_05', 'demo_aluno_06');

DELETE FROM students
WHERE id IN ('demo_aluno_01', 'demo_aluno_02', 'demo_aluno_03', 'demo_aluno_04', 'demo_aluno_05', 'demo_aluno_06');

DELETE FROM teachers
WHERE id IN (8001, 8002);

DELETE FROM users
WHERE cpf IN ('12345678909', '52998224725');

DELETE FROM subjects
WHERE id IN (7001, 7002);

DELETE FROM classes
WHERE id IN (6001, 6002);

COMMIT;
