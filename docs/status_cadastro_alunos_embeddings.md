# Status do cadastro de alunos e embeddings faciais

Atualizado em 2026-05-26.

## Rotas criadas/ajustadas

- `GET /api/students`
- `POST /api/students`
- `GET /api/students/{student_id}`
- `PUT /api/students/{student_id}`
- `PATCH /api/students/{student_id}`
- `DELETE /api/students/{student_id}`
- `POST /api/students/{student_id}/face-images`
- `POST /api/students/{student_id}/generate-embeddings`
- `GET /api/students/{student_id}/face-status`

Rotas auxiliares mantidas:

- `GET /api/students/classes`
- `GET /api/students/next-registration`

## Componentes criados

- `frontend/src/components/students/StudentForm.tsx`
- `frontend/src/components/students/StudentTable.tsx`
- `frontend/src/components/students/StudentProfileCard.tsx`
- `frontend/src/components/students/FaceCaptureWizard.tsx`
- `frontend/src/components/students/PoseProgressCard.tsx`
- `frontend/src/components/students/CameraCapturePanel.tsx`
- `frontend/src/components/students/EmbeddingStatusCard.tsx`

## Arquivos de suporte criados/alterados

- `backend/app/services/face_capture_service.py`
- `backend/app/services/embedding_service.py`
- `backend/app/api/routes_students.py`
- `backend/app/schemas/students.py`
- `frontend/src/pages/Students.tsx`
- `frontend/src/hooks/useStudents.ts`
- `frontend/src/hooks/useFaceCapture.ts`
- `frontend/src/services/studentsApi.ts`
- `frontend/src/services/api.ts`
- `frontend/src/types/student.ts`
- `frontend/src/types/students.ts`

## Fluxo implementado

1. O usuário acessa a tela `Alunos`.
2. Preenche nome, matrícula, turma/classe e observações locais.
3. Salva o aluno pela API.
4. Seleciona o aluno salvo na lista ou usa o aluno recém-criado.
5. O wizard orienta as poses:
   - `frontal`
   - `lateral_esquerda`
   - `lateral_direita`
   - `cabeca_baixa`
6. A captura usa webcam local do navegador com `getUserMedia`.
7. As imagens capturadas são enviadas para `POST /api/students/{student_id}/face-images`.
8. O backend salva as imagens em `data/alunos/<student_id>/<pose>/...`.
9. O status facial mostra quantidade por pose e se a pose está completa.
10. Com as quatro poses completas, o botão `Gerar embeddings` é liberado.
11. A geração chama a lógica legada `generate_student_embedding` de `src/register_face_multi_images_avg.py`.
12. O status mostra se o aluno está pronto para reconhecimento.

## Como testar cadastro

1. Inicie o backend:

```bash
PYTHONPATH=backend .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Inicie o frontend:

```bash
cd frontend
npm run dev
```

3. Acesse a tela `Alunos`.
4. Cadastre nome, matrícula e turma.
5. Confirme se o aluno aparece na lista.

Via API:

```bash
curl -X POST http://localhost:8000/api/students \
  -H "Authorization: Bearer SEU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Aluno Teste","matricula":"MAT2026001","class_id":1}'
```

## Como testar captura

No frontend:

1. Selecione um aluno.
2. Permita acesso à webcam quando o navegador solicitar.
3. Capture imagens na pose atual.
4. Clique em `Enviar capturas`.
5. Avance para a próxima pose até completar as quatro poses.

Via API, para enviar imagens de uma pose:

```bash
curl -X POST http://localhost:8000/api/students/STUDENT_ID/face-images \
  -H "Authorization: Bearer SEU_TOKEN" \
  -F "pose=frontal" \
  -F "files=@imagem1.jpg" \
  -F "files=@imagem2.jpg"
```

## Como testar geração de embeddings

Após capturar imagens suficientes nas quatro poses:

```bash
curl -X POST http://localhost:8000/api/students/STUDENT_ID/generate-embeddings \
  -H "Authorization: Bearer SEU_TOKEN"
```

Consultar status:

```bash
curl http://localhost:8000/api/students/STUDENT_ID/face-status \
  -H "Authorization: Bearer SEU_TOKEN"
```

Resposta esperada:

```json
{
  "student_id": "hash_do_aluno",
  "frontal": {"count": 10, "complete": true},
  "lateral_esquerda": {"count": 10, "complete": true},
  "lateral_direita": {"count": 10, "complete": true},
  "cabeca_baixa": {"count": 10, "complete": true},
  "embeddings_generated": true
}
```

## Segurança e LGPD

- O frontend não recebe paths internos do servidor.
- O frontend não recebe embeddings.
- As imagens são salvas sob `data/alunos/<student_id>/<pose>` seguindo a estrutura atual do legado.
- O identificador do aluno continua sendo hash/pseudônimo derivado de nome e matrícula.
- A limpeza automática das imagens brutas ainda não foi implementada nesta etapa porque o fluxo legado atual usa essas imagens como inventário operacional para reprocessamento.

## O que ainda falta

- Captura automática de 10 imagens com intervalo, como no Streamlit.
- Validação de qualidade da imagem antes do upload.
- Feedback mais detalhado de erro do InsightFace no frontend.
- Painel administrativo para reprocessar embeddings pendentes em lote.
- Política configurável para descarte ou retenção de imagens brutas após gerar embeddings.
- Testes automatizados específicos para upload multipart e status facial.

## Próximos passos

1. Validar webcam local em navegador real com HTTPS/local dev.
2. Testar geração de embeddings no ambiente com `insightface`, `onnxruntime` e `opencv` instalados.
3. Conectar status de embeddings ao monitoramento em tempo real.
4. Definir política LGPD de retenção de imagens.
5. Adicionar captura automática guiada por pose, reaproveitando os tempos do legado.

## Situação atual

A tela `Alunos` já suporta cadastro, seleção de aluno, captura facial por quatro poses, upload para o backend, consulta de status por pose e acionamento da geração de embeddings pela lógica legada. O Streamlit permanece intacto.
