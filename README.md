# Sistema de Monitoramento Comportamental em Sala

Sistema com **Streamlit + YOLO Pose + InsightFace** para:
- cadastrar alunos com fotos por pose,
- gerar embeddings faciais,
- monitorar comportamentos em tempo real,
- salvar episódios de comportamento no PostgreSQL,
- exibir gráficos de tempo por comportamento.

## 1. O que o sistema salva

### Arquivos locais (`data/`)
- `data/mapeamento_alunos.csv`: cadastro lógico (`nome`, `matricula`, `hash`).
- `data/alunos/<hash>/...`: fotos por pose de cada aluno.
- `data/embeddings.npy`: embeddings médios por aluno (fallback local).
- `data/names.pkl`: nomes na ordem dos embeddings (fallback local).
- `data/behavior_episodes_fake_YYYYMMDD.csv`: CSV de episódios simulados (quando gerar seed).

### Banco PostgreSQL
- `users`: usuários da aplicação (login).
- `students`: referência básica de alunos (`id/hash`, `name`).
- `face_embeddings`: embeddings persistidos no banco.
- `behavior_episode`: episódios de comportamento (tempo).
- `behavior_log`: legado (não usado pelos gráficos novos).

## 2. Fluxo de arquitetura (resumo)

1. **Cadastro** (UI) grava fotos + atualiza `mapeamento_alunos.csv`.
2. **Script de embeddings** lê fotos e grava `embeddings.npy`/`names.pkl` + `face_embeddings` no banco.
3. **Monitoramento** reconhece aluno e classifica comportamento.
4. Com debounce, o sistema fecha/abre episódios e grava em `behavior_episode`.
5. **Gráficos** leem `behavior_episode` (minutos e percentuais).

## 3. Pré-requisitos

- Docker + Docker Compose
- (Opcional) GPU CUDA para acelerar IA
- Fonte de vídeo:
  - RTSP (`rtsp://...`) ou
  - webcam local (`/dev/video0` no Linux)

## 4. Configuração inicial

### 4.1 Ajuste `.env`
Campos principais:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `RTSP_URL`
- `RELAY_PORT`, `RELAY_SEND_FPS`, `RELAY_QUALITY`

Observação importante:
- Fora do Docker: normalmente `DB_HOST=localhost`
- Dentro do Compose (serviço `app`): o host do banco é `db`

### 4.2 Suba os serviços
```bash
docker compose up --build -d db relay app
```
Observação:
- O compose principal já sobe o serviço `app` com GPU NVIDIA (`gpus: all`).
- Isso exige host com driver NVIDIA + NVIDIA Container Toolkit configurados.
- Se a máquina não tiver GPU NVIDIA disponível, o serviço `app` pode falhar ao iniciar.

### 4.3 Verifique logs
```bash
docker compose logs -f db
docker compose logs -f relay
docker compose logs -f app
```

## 5. Procedimento operacional completo (para quem nunca usou)

## 5.1 Acessar aplicação
- URL padrão: `http://localhost:8501`
- Faça login/cadastro de usuário se necessário.

## 5.2 Cadastrar alunos (fotos por pose)
No menu **Cadastro de Alunos**:
1. Preencha `Nome` e `Matrícula`.
2. Configure quantidade/intervalo de captura.
3. Capture as poses guiadas até finalizar.

O sistema vai:
- gerar/atualizar `hash` do aluno,
- atualizar `data/mapeamento_alunos.csv`,
- salvar fotos em `data/alunos/<hash>/<pose>/...`.

## 5.3 Gerar embeddings após cadastro (obrigatório)
Depois de cadastrar/alterar alunos, execute:
```bash
docker compose run --rm app python src/register_face_multi_images_avg.py
```

Esse script vai:
1. Ler `data/mapeamento_alunos.csv`.
2. Processar imagens por aluno.
3. Gerar embedding médio por aluno.
4. Atualizar arquivos locais:
   - `data/embeddings.npy`
   - `data/names.pkl`
5. Fazer upsert em `face_embeddings` no PostgreSQL.

## 5.4 (Opcional) Popular dados simulados para gráficos
Para demos/testes sem monitoramento ao vivo:
```bash
docker compose run --rm app python src/seed_existing_students_and_fake_behavior.py --date 2026-02-13 --seed 42 --total-minutes 50
```

Esse script vai:
- ler alunos do `mapeamento_alunos.csv`,
- fazer upsert em `students`,
- reforçar/upsert em `face_embeddings` (se houver embeddings locais),
- gerar episódios simulados,
- inserir em `behavior_episode`,
- exportar CSV `data/behavior_episodes_fake_20260213.csv`.

Comportamentos simulados atuais:
- `Atento`, `Perguntando`, `Dormindo`, `Distraido`, `Agitado`.

## 5.5 Monitoramento em tempo real
No menu **Monitoramento**:
1. Garanta que o relay está ativo.
2. Clique **Iniciar Monitoramento**.
3. Ajuste confiança/GPU conforme ambiente.

Durante monitoramento, o sistema:
- detecta/identifica aluno,
- classifica comportamento,
- aplica debounce de estabilidade,
- salva **episódios** em `behavior_episode` (não salva por frame).

## 5.6 Gráficos
No menu **Gráficos**:
- filtro por aluno, disciplina e data,
- gráficos de:
  - percentual do tempo por comportamento,
  - tempo total (minutos) por comportamento,
  - linha do tempo de episódios.

Download:
- PNG via ícone de câmera do Plotly (nativo no browser).
- Download consolidado via backend depende do ambiente (quando disponível).

## 6. Quando cada tabela é persistida

- `users`: no cadastro/login de usuário.
- `students`: no script de seed (`seed_existing_students_and_fake_behavior.py`).
- `face_embeddings`:
  - no `register_face_multi_images_avg.py` (principal),
  - também no seed, quando houver embeddings locais.
- `behavior_episode`:
  - no monitoramento ao vivo (episódios reais, `source=realtime`),
  - no seed (episódios simulados, `source=simulated`).

## 7. Validação rápida (SQL)

```sql
SELECT COUNT(*) FROM students;
SELECT COUNT(*) FROM face_embeddings;
SELECT COUNT(*) FROM behavior_episode;

SELECT source, COUNT(*)
FROM behavior_episode
GROUP BY source;
```

## 8. Comandos úteis

### Subir stack
```bash
docker compose up -d db relay app
```

### Rodar embeddings
```bash
docker compose run --rm app python src/register_face_multi_images_avg.py
```

### Gerar simulados
```bash
docker compose run --rm app python src/seed_existing_students_and_fake_behavior.py --date 2026-02-13 --seed 42 --total-minutes 50
```

### Entrar no banco e validar
```bash
docker compose exec -T db psql -U insightface_user -d insightface_db
```

## 9. Troubleshooting

### `students` aparece vazio no Database Navigator
- Verifique se está na conexão/banco corretos.
- Faça refresh de schema/tabela.
- Rode `SELECT COUNT(*) FROM students;`.
- Confirme se o seed foi executado no mesmo banco do app.

### Aluno aparece como desconhecido
- Reexecute o script de embeddings após novos cadastros.
- Garanta fotos frontais com iluminação adequada.
- Verifique se `face_embeddings` foi realmente populada.

### Gráfico sem dados
- Confira filtros (aluno/disciplina/data).
- Valide se há linhas em `behavior_episode` para a data.

### Delay no vídeo
- Reduza `RELAY_SEND_FPS`.
- Use stream secundário da câmera (menor resolução/bitrate).
- Evite múltiplos consumidores simultâneos do mesmo RTSP.

## 10. Boas práticas operacionais

- Após qualquer novo cadastro: **sempre** rodar geração de embeddings.
- Para demos: gerar dados simulados antes da apresentação.
- Não misturar conexões de banco local e do Docker sem confirmar `DB_HOST`.
- Monitorar logs de `relay` e `app` quando houver falhas de detecção.
