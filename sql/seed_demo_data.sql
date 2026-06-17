BEGIN;

INSERT INTO users (id, cpf, nome, password, cidade, estado, role, ativo)
VALUES
    (2001, '12345678909', 'Prof. Ana Lima', '$2b$12$FwfOcVuxtKcrxCICQnE75.DBVHS0c2GHqgA23bI0Qx/W63HfN8Lwi', 'Palmas', 'TO', 'professor', TRUE),
    (2002, '52998224725', 'Prof. Bruno Costa', '$2b$12$3XPVO7CwZstWu7lxi6c8rOSby16p.ABKNQIjdSwET7gYha9OQBb52', 'Palmas', 'TO', 'professor', TRUE)
ON CONFLICT (cpf)
DO UPDATE SET
    nome = EXCLUDED.nome,
    password = EXCLUDED.password,
    cidade = EXCLUDED.cidade,
    estado = EXCLUDED.estado,
    role = EXCLUDED.role,
    ativo = EXCLUDED.ativo;

INSERT INTO teachers (id, user_id, nome)
VALUES
    (8001, 2001, 'Prof. Ana Lima'),
    (8002, 2002, 'Prof. Bruno Costa')
ON CONFLICT (user_id)
DO UPDATE SET
    nome = EXCLUDED.nome;

INSERT INTO subjects (id, nome)
VALUES
    (7001, 'Matemática'),
    (7002, 'Português')
ON CONFLICT (id)
DO UPDATE SET
    nome = EXCLUDED.nome;

INSERT INTO classes (id, nome, identificador)
VALUES
    (6001, 'Turma A', '1A-2026'),
    (6002, 'Turma B', '2A-2026')
ON CONFLICT (id)
DO UPDATE SET
    nome = EXCLUDED.nome,
    identificador = EXCLUDED.identificador;

INSERT INTO teacher_subject_class (teacher_id, subject_id, class_id)
VALUES
    (8001, 7001, 6001),
    (8001, 7002, 6001),
    (8002, 7001, 6002),
    (8002, 7002, 6002)
ON CONFLICT (teacher_id, subject_id, class_id)
DO NOTHING;

INSERT INTO students (id, name, matricula, class_id, ativo)
VALUES
    ('demo_aluno_01', 'Alice Souza', 'MAT2026001', 6001, TRUE),
    ('demo_aluno_02', 'Beatriz Rocha', 'MAT2026002', 6001, TRUE),
    ('demo_aluno_03', 'Carlos Melo', 'MAT2026003', 6001, TRUE),
    ('demo_aluno_04', 'Daniela Alves', 'MAT2026004', 6002, TRUE),
    ('demo_aluno_05', 'Eduardo Nunes', 'MAT2026005', 6002, TRUE),
    ('demo_aluno_06', 'Fernanda Luz', 'MAT2026006', 6002, TRUE)
ON CONFLICT (id)
DO UPDATE SET
    name = EXCLUDED.name,
    matricula = EXCLUDED.matricula,
    class_id = EXCLUDED.class_id,
    ativo = EXCLUDED.ativo;

INSERT INTO monitoring_sessions (id, teacher_id, subject_id, class_id, lesson_type, session_date, start_time, end_time, status)
VALUES
    (9001, 8001, 7001, 6001, 'Atividade', DATE '2026-03-20', TIMESTAMP '2026-03-20 08:00:00', TIMESTAMP '2026-03-20 08:50:00', 'encerrada'),
    (9002, 8001, 7002, 6001, 'Prova', DATE '2026-03-21', TIMESTAMP '2026-03-21 09:00:00', TIMESTAMP '2026-03-21 09:50:00', 'encerrada'),
    (9003, 8001, 7001, 6001, 'Atividade', DATE '2026-03-24', TIMESTAMP '2026-03-24 10:00:00', TIMESTAMP '2026-03-24 10:50:00', 'encerrada'),
    (9004, 8002, 7001, 6002, 'Prova', DATE '2026-03-20', TIMESTAMP '2026-03-20 13:00:00', TIMESTAMP '2026-03-20 13:50:00', 'encerrada'),
    (9005, 8002, 7002, 6002, 'Atividade', DATE '2026-03-22', TIMESTAMP '2026-03-22 14:00:00', TIMESTAMP '2026-03-22 14:50:00', 'encerrada'),
    (9006, 8002, 7001, 6002, 'Prova', DATE '2026-03-25', TIMESTAMP '2026-03-25 15:00:00', TIMESTAMP '2026-03-25 15:50:00', 'encerrada')
ON CONFLICT (id)
DO UPDATE SET
    teacher_id = EXCLUDED.teacher_id,
    subject_id = EXCLUDED.subject_id,
    class_id = EXCLUDED.class_id,
    lesson_type = EXCLUDED.lesson_type,
    session_date = EXCLUDED.session_date,
    start_time = EXCLUDED.start_time,
    end_time = EXCLUDED.end_time,
    status = EXCLUDED.status;

INSERT INTO behavior_episode (
    monitoring_session_id, student_id, lesson_type, school, discipline, teacher, student, id_student,
    behavior, start_time, end_time, duration_seconds, date, source
)
VALUES
    (9001, 'demo_aluno_01', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Alice Souza', 'demo_aluno_01', 'Atento', TIMESTAMP '2026-03-20 08:02:00', TIMESTAMP '2026-03-20 08:14:00', 720, DATE '2026-03-20', 'seed_demo'),
    (9001, 'demo_aluno_01', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Alice Souza', 'demo_aluno_01', 'Perguntando', TIMESTAMP '2026-03-20 08:18:00', TIMESTAMP '2026-03-20 08:20:00', 120, DATE '2026-03-20', 'seed_demo'),
    (9001, 'demo_aluno_02', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Beatriz Rocha', 'demo_aluno_02', 'Distraido', TIMESTAMP '2026-03-20 08:10:00', TIMESTAMP '2026-03-20 08:16:00', 360, DATE '2026-03-20', 'seed_demo'),
    (9001, 'demo_aluno_03', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Carlos Melo', 'demo_aluno_03', 'Escrevendo', TIMESTAMP '2026-03-20 08:21:00', TIMESTAMP '2026-03-20 08:34:00', 780, DATE '2026-03-20', 'seed_demo'),

    (9002, 'demo_aluno_01', 'Prova', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Ana Lima', 'Alice Souza', 'demo_aluno_01', 'Atento', TIMESTAMP '2026-03-21 09:05:00', TIMESTAMP '2026-03-21 09:17:00', 720, DATE '2026-03-21', 'seed_demo'),
    (9002, 'demo_aluno_02', 'Prova', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Ana Lima', 'Beatriz Rocha', 'demo_aluno_02', 'Escrevendo', TIMESTAMP '2026-03-21 09:18:00', TIMESTAMP '2026-03-21 09:28:00', 600, DATE '2026-03-21', 'seed_demo'),
    (9002, 'demo_aluno_03', 'Prova', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Ana Lima', 'Carlos Melo', 'demo_aluno_03', 'Agitado', TIMESTAMP '2026-03-21 09:30:00', TIMESTAMP '2026-03-21 09:34:00', 240, DATE '2026-03-21', 'seed_demo'),
    (9002, 'demo_aluno_03', 'Prova', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Ana Lima', 'Carlos Melo', 'demo_aluno_03', 'Atento', TIMESTAMP '2026-03-21 09:35:00', TIMESTAMP '2026-03-21 09:46:00', 660, DATE '2026-03-21', 'seed_demo'),

    (9003, 'demo_aluno_01', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Alice Souza', 'demo_aluno_01', 'Atento', TIMESTAMP '2026-03-24 10:03:00', TIMESTAMP '2026-03-24 10:18:00', 900, DATE '2026-03-24', 'seed_demo'),
    (9003, 'demo_aluno_02', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Beatriz Rocha', 'demo_aluno_02', 'Dormindo', TIMESTAMP '2026-03-24 10:22:00', TIMESTAMP '2026-03-24 10:27:00', 300, DATE '2026-03-24', 'seed_demo'),
    (9003, 'demo_aluno_03', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Carlos Melo', 'demo_aluno_03', 'Distraido', TIMESTAMP '2026-03-24 10:28:00', TIMESTAMP '2026-03-24 10:33:00', 300, DATE '2026-03-24', 'seed_demo'),
    (9003, 'demo_aluno_03', 'Atividade', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Ana Lima', 'Carlos Melo', 'demo_aluno_03', 'Atento', TIMESTAMP '2026-03-24 10:34:00', TIMESTAMP '2026-03-24 10:44:00', 600, DATE '2026-03-24', 'seed_demo'),

    (9004, 'demo_aluno_04', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Daniela Alves', 'demo_aluno_04', 'Atento', TIMESTAMP '2026-03-20 13:02:00', TIMESTAMP '2026-03-20 13:16:00', 840, DATE '2026-03-20', 'seed_demo'),
    (9004, 'demo_aluno_05', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Eduardo Nunes', 'demo_aluno_05', 'Perguntando', TIMESTAMP '2026-03-20 13:18:00', TIMESTAMP '2026-03-20 13:21:00', 180, DATE '2026-03-20', 'seed_demo'),
    (9004, 'demo_aluno_06', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Fernanda Luz', 'demo_aluno_06', 'Escrevendo', TIMESTAMP '2026-03-20 13:22:00', TIMESTAMP '2026-03-20 13:37:00', 900, DATE '2026-03-20', 'seed_demo'),

    (9005, 'demo_aluno_04', 'Atividade', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Bruno Costa', 'Daniela Alves', 'demo_aluno_04', 'Atento', TIMESTAMP '2026-03-22 14:01:00', TIMESTAMP '2026-03-22 14:15:00', 840, DATE '2026-03-22', 'seed_demo'),
    (9005, 'demo_aluno_05', 'Atividade', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Bruno Costa', 'Eduardo Nunes', 'demo_aluno_05', 'Distraido', TIMESTAMP '2026-03-22 14:16:00', TIMESTAMP '2026-03-22 14:20:00', 240, DATE '2026-03-22', 'seed_demo'),
    (9005, 'demo_aluno_06', 'Atividade', 'Escola Estadual Criança Esperança', 'Português', 'Prof. Bruno Costa', 'Fernanda Luz', 'demo_aluno_06', 'Atento', TIMESTAMP '2026-03-22 14:21:00', TIMESTAMP '2026-03-22 14:36:00', 900, DATE '2026-03-22', 'seed_demo'),

    (9006, 'demo_aluno_04', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Daniela Alves', 'demo_aluno_04', 'Escrevendo', TIMESTAMP '2026-03-25 15:05:00', TIMESTAMP '2026-03-25 15:18:00', 780, DATE '2026-03-25', 'seed_demo'),
    (9006, 'demo_aluno_05', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Eduardo Nunes', 'demo_aluno_05', 'Agitado', TIMESTAMP '2026-03-25 15:20:00', TIMESTAMP '2026-03-25 15:24:00', 240, DATE '2026-03-25', 'seed_demo'),
    (9006, 'demo_aluno_06', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Fernanda Luz', 'demo_aluno_06', 'Dormindo', TIMESTAMP '2026-03-25 15:28:00', TIMESTAMP '2026-03-25 15:33:00', 300, DATE '2026-03-25', 'seed_demo'),
    (9006, 'demo_aluno_06', 'Prova', 'Escola Estadual Criança Esperança', 'Matemática', 'Prof. Bruno Costa', 'Fernanda Luz', 'demo_aluno_06', 'Atento', TIMESTAMP '2026-03-25 15:34:00', TIMESTAMP '2026-03-25 15:46:00', 720, DATE '2026-03-25', 'seed_demo')
ON CONFLICT DO NOTHING;

SELECT setval('users_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM users), 1), TRUE);
SELECT setval('teachers_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM teachers), 1), TRUE);
SELECT setval('subjects_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM subjects), 1), TRUE);
SELECT setval('classes_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM classes), 1), TRUE);
SELECT setval('teacher_subject_class_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM teacher_subject_class), 1), TRUE);
SELECT setval('monitoring_sessions_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM monitoring_sessions), 1), TRUE);
SELECT setval('behavior_episode_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM behavior_episode), 1), TRUE);

COMMIT;
