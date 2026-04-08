# Sistema de Monitoramento Comportamental em Sala

Sistema com `Streamlit + InsightFace + YOLO Pose + PostgreSQL/pgvector` para cadastro de alunos, reconhecimento facial, monitoramento comportamental em tempo real, geração de gráficos analíticos e relatórios observacionais.

## 1. Visão geral do que o sistema já faz

O sistema foi evoluído para cobrir o fluxo operacional completo de uso em ambiente escolar:

- autenticação de usuários com perfis `professor` e `admin`;
- persistência de sessão com cookie e fallback por token assinado na URL;
- cadastro de alunos com captura guiada por poses;
- associação do aluno a uma turma;
- geração automática ou reprocessamento de embeddings faciais;
- armazenamento local dos vetores e sincronização no PostgreSQL com `pgvector`;
- criação de sessões formais de monitoramento por professor, disciplina, turma e tipo de aula;
- reconhecimento facial durante o monitoramento;
- classificação comportamental com regras de estabilidade para reduzir falsos positivos;
- persistência por episódio comportamental, em vez de salvar frame a frame;
- gráficos por aluno com filtros por professor, disciplina, turma e período;
- comparação dos comportamentos por tipo de aula;
- relatórios observacionais com sínteses textuais, tabelas e exportações;
- painel administrativo para usuários e manutenção de embeddings.

## 2. Funcionalidades importantes

### 2.1 Login, autenticação e sessão

O acesso à aplicação é protegido por autenticação com CPF e senha.

Principais pontos:

- a tabela `users` é criada e validada automaticamente na inicialização;
- a senha é armazenada com `bcrypt`;
- a sessão do usuário é restaurada com cookie;
- existe fallback opcional por token assinado em query string para lidar com falhas de restauração do Streamlit após refresh;
- o sistema diferencia usuários `professor` e `admin`;
- o logout limpa cookie, parâmetros legados, estado da sessão e contexto de monitoramento.

Na prática, isso reduz perda de sessão e melhora a confiabilidade da navegação entre telas.

### 2.2 Perfis de acesso

O comportamento da interface muda conforme o perfil:

- `professor`: acessa `Cadastro de Alunos`, `Monitoramento`, `Gráficos` e `Relatórios`;
- `admin`: acessa `Usuários`, `Gráficos` e `Relatórios`.

Além disso:

- professores só veem turmas, disciplinas e alunos dentro do próprio escopo;
- administradores podem visualizar o conjunto completo dos dados disponíveis no banco.

### 2.3 Cadastro de alunos com captura guiada

O cadastro de alunos não é apenas um formulário simples. Ele foi atualizado para operar com fluxo guiado de captura de imagens.

O professor pode:

- selecionar a turma;
- indicar se o aluno é novo ou já existente;
- aproveitar um aluno já cadastrado para atualizar poses e imagens;
- usar matrícula gerada automaticamente quando o cadastro for novo;
- avançar por um processo de captura assistida.

Poses atualmente usadas no cadastro:

- `frontal`
- `lateral_direita`
- `lateral_esquerda`
- `cabeca_baixa`

Durante esse processo o sistema:

- valida nome e matrícula;
- gera ou reutiliza o identificador lógico do aluno;
- grava imagens em `data/alunos/<hash>/<pose>/...`;
- atualiza `data/mapeamento_alunos.csv`;
- pode associar o aluno à turma selecionada no banco.

### 2.4 Geração de embeddings faciais

Uma das atualizações mais importantes foi a consolidação do fluxo de embeddings.

O sistema agora:

- lê o mapeamento de alunos em `data/mapeamento_alunos.csv`;
- percorre todas as imagens disponíveis por pose;
- detecta rostos válidos nas fotos;
- calcula um embedding médio por aluno;
- salva o vetor localmente em `data/embeddings.npy`;
- salva a ordem dos nomes em `data/names.pkl`;
- faz `upsert` do embedding na tabela `face_embeddings`.

Isso traz dois níveis de persistência:

- fallback local para reconhecimento e manutenção;
- persistência estruturada no PostgreSQL usando `vector(512)`.

### 2.5 Geração automática de embedding após cadastro

O fluxo de cadastro foi melhorado para tentar concluir o processamento facial logo após a captura.

Ao finalizar um cadastro:

- o sistema tenta gerar o embedding daquele aluno;
- informa se a sincronização com o banco foi bem-sucedida;
- permite reprocessar manualmente o embedding do aluno recém-cadastrado;
- atualiza o estado interno do monitoramento para que o novo aluno possa ser reconhecido sem inconsistências.

Isso reduz o risco de cadastrar o aluno e esquecer de preparar o reconhecimento facial.

### 2.6 Painel de manutenção de embeddings

O painel administrativo ganhou uma área específica para acompanhar a saúde dos embeddings.

Esse painel mostra:

- quantidade total de alunos mapeados;
- quantos embeddings estão atualizados;
- quantos estão pendentes;
- quantidade de imagens disponíveis por aluno;
- diagnóstico do motivo de pendência.

Ações disponíveis:

- `Reprocessar embeddings pendentes`
- `Reprocessar todos os embeddings`

Essa funcionalidade é importante porque facilita operação e suporte sem depender sempre de linha de comando.

### 2.7 Administração de usuários

O sistema já possui um módulo administrativo para gestão de acesso.

O administrador pode:

- listar usuários cadastrados;
- criar novos usuários;
- definir perfil `professor` ou `admin`;
- registrar CPF, nome, cidade, estado e email;
- redefinir senha de usuários existentes.

No cadastro:

- o CPF é validado;
- a senha é confirmada antes do salvamento;
- o perfil de professor gera vínculo de professor no banco quando necessário.

### 2.8 Estrutura acadêmica no banco

O banco foi ampliado para refletir melhor o contexto escolar real.

Além de `users`, `students` e `face_embeddings`, o schema agora contempla:

- `teachers`
- `subjects`
- `classes`
- `teacher_subject_class`
- `monitoring_sessions`
- `behavior_episode`

Isso permite que o sistema trabalhe com:

- professor responsável;
- disciplina da aula;
- turma acompanhada;
- tipo de aula;
- sessão de monitoramento aberta e encerrada formalmente.

### 2.9 Sessões de monitoramento

O monitoramento em tempo real deixou de ser apenas uma tela de vídeo e passou a registrar sessões completas.

Ao iniciar o monitoramento, o professor seleciona:

- disciplina da aula;
- turma acompanhada;
- tipo de aula.

Quando a sessão começa:

- a tabela `monitoring_sessions` recebe uma nova linha;
- a sessão fica com status `em_andamento`;
- o sistema monta um lookup dos alunos do escopo permitido;
- o gerenciador de episódios comportamentais é inicializado;
- o runtime do monitoramento é preparado com a fonte de vídeo configurada.

Ao encerrar:

- a sessão recebe `end_time`;
- o status muda para `encerrada`;
- os episódios ficam vinculados à sessão monitorada.

### 2.10 Reconhecimento facial e monitoramento em tempo real

Durante o monitoramento, o sistema:

- consome vídeo via relay/RTSP;
- detecta pessoas e rostos;
- tenta identificar o aluno com base nos embeddings;
- aplica classificação comportamental;
- exibe o resumo visual da sessão.

O reconhecimento considera o escopo da turma do professor, o que ajuda a reduzir ambiguidades com alunos fora da sessão.

### 2.11 Classificação comportamental com regras mais conservadoras

O classificador comportamental foi refinado para evitar decisões otimistas demais.

Comportamentos mapeados no sistema:

- `Atento`
- `Distraido`
- `Perguntando`
- `Dormindo`
- `Agitado`
- `Escrevendo`
- `Em Pé`

Pontos importantes do comportamento:

- `Atento` não é mais fallback automático;
- estados ambíguos tendem a permanecer mais conservadores;
- cabeça baixa passa a bloquear promoção indevida para `Atento`;
- `Perguntando` exige evidência mais consistente de mão levantada;
- a troca de estado usa critérios de estabilidade em frames e tempo.

Na prática, isso melhora a qualidade do dado salvo.

### 2.12 Persistência por episódio comportamental

Uma atualização central do sistema foi passar a persistir comportamento por episódio, e não por frame.

Em vez de salvar milhares de eventos instantâneos, o sistema:

- abre um episódio quando o comportamento estabiliza;
- mantém o episódio enquanto o estado continua consistente;
- encerra o episódio quando há mudança estável;
- grava `start_time`, `end_time`, `duration_seconds`, `student`, `student_id`, `monitoring_session_id`, `lesson_type` e `source`.

Benefícios:

- dados mais limpos;
- consultas mais rápidas;
- gráficos mais interpretáveis;
- relatórios mais fiéis ao tempo real observado.

### 2.13 Gráficos comportamentais

O sistema já possui uma área de gráficos com filtros e resumos analíticos.

Filtros disponíveis conforme escopo do usuário:

- professor;
- disciplina;
- turma;
- aluno;
- intervalo de datas.

Os gráficos e análises incluem:

- resumo textual automático do comportamento predominante;
- tempo total por comportamento;
- percentual do tempo por comportamento;
- linha do tempo dos episódios;
- comparação por tipo de aula;
- exportação da base consolidada usada nos gráficos.

Os dados usados nessa área vêm principalmente de `behavior_episode` associado a `monitoring_sessions`.

### 2.14 Relatórios observacionais

Além dos gráficos, o sistema já possui uma camada de relatórios mais interpretativos.

A página de relatórios:

- gera consolidação por aluno;
- considera professor, disciplina, turma e período;
- monta indicadores principais;
- calcula distribuição diária, por faixa horária e por segmento da aula;
- compara com período anterior equivalente;
- mede consistência de cada comportamento;
- identifica dias de pico;
- produz sínteses textuais e notas metodológicas;
- suporta exportações em CSV e PDF.

É uma camada mais adequada para análise pedagógica do que apenas olhar gráficos brutos.

### 2.15 Dados simulados para demonstração

O projeto inclui suporte para geração de dados simulados.

O script de seed:

- lê os alunos cadastrados;
- faz `upsert` em `students`;
- reaproveita embeddings locais quando disponíveis;
- cria episódios simulados em `behavior_episode`;
- exporta CSV em `data/behavior_episodes_fake_YYYYMMDD.csv`.

Isso ajuda em:

- testes sem câmera ativa;
- validação de gráficos;
- demonstrações para apresentação;
- homologação de relatórios.

## 3. O que o sistema salva

### 3.1 Arquivos locais em `data/`

- `data/mapeamento_alunos.csv`: cadastro lógico com `nome`, `matricula` e `hash`;
- `data/alunos/<hash>/...`: imagens do aluno organizadas por pose;
- `data/embeddings.npy`: matriz local de embeddings;
- `data/names.pkl`: nomes na ordem dos embeddings locais;
- `data/behavior_episodes_fake_YYYYMMDD.csv`: exportações de episódios simulados.

### 3.2 Tabelas relevantes no PostgreSQL

- `users`: autenticação e perfil do usuário;
- `teachers`: vínculo de usuário professor;
- `subjects`: disciplinas;
- `classes`: turmas;
- `teacher_subject_class`: escopo professor-disciplina-turma;
- `students`: alunos e associação com turma;
- `face_embeddings`: vetores faciais persistidos;
- `monitoring_sessions`: sessões formais de monitoramento;
- `behavior_episode`: episódios comportamentais reais ou simulados;
- `behavior_log`: estrutura legada, mantida para compatibilidade.

## 4. Fluxo operacional resumido

1. O usuário faz login.
2. O professor acessa `Cadastro de Alunos`.
3. Seleciona turma e cadastra ou atualiza um aluno.
4. O sistema captura imagens por pose.
5. O embedding do aluno é gerado e sincronizado.
6. O professor abre uma sessão em `Monitoramento`.
7. Seleciona disciplina, turma e tipo de aula.
8. O sistema reconhece o aluno e classifica o comportamento.
9. Os episódios são persistidos em `behavior_episode`.
10. `Gráficos` e `Relatórios` consomem esses episódios para análise.

## 5. Pré-requisitos

- Docker + Docker Compose;
- PostgreSQL;
- extensão `pgvector`;
- câmera local ou stream RTSP;
- GPU CUDA opcional para acelerar processamento.

## 6. Configuração inicial

### 6.1 Variáveis de ambiente principais

Campos importantes do `.env`:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `RTSP_URL`
- `RELAY_PORT`
- `RELAY_SEND_FPS`
- `RELAY_QUALITY`
- `APP_TIMEZONE`
- `AUTH_ENABLE_QUERY_TOKEN`
- `AUTH_TOKEN_SECRET`

Observações:

- fora do Docker, normalmente o banco fica em `localhost`;
- dentro do `docker compose`, o host do banco costuma ser `db`.

### 6.2 Subir os serviços

```bash
docker compose up --build -d db relay app
```

### 6.3 Verificar logs

```bash
docker compose logs -f db
docker compose logs -f relay
docker compose logs -f app
```

## 7. Comandos úteis

### 7.1 Reprocessar embeddings via CLI

```bash
docker compose run --rm app python src/register_face_multi_images_avg.py
```

### 7.2 Gerar dados simulados

```bash
docker compose run --rm app python src/seed_existing_students_and_fake_behavior.py --date 2026-02-13 --seed 42 --total-minutes 50
```

### 7.3 Entrar no banco

```bash
docker compose exec -T db psql -U insightface_user -d insightface_db
```

## 8. Validação rápida no banco

```sql
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM students;
SELECT COUNT(*) FROM face_embeddings;
SELECT COUNT(*) FROM monitoring_sessions;
SELECT COUNT(*) FROM behavior_episode;

SELECT source, COUNT(*)
FROM behavior_episode
GROUP BY source;
```

## 9. Troubleshooting

### 9.1 Aluno aparece como desconhecido

- confirme se o cadastro do aluno foi concluído com embedding gerado;
- use o painel de manutenção de embeddings para verificar pendências;
- reprocesse o embedding do aluno ou todos os pendentes;
- confira se existem imagens válidas nas pastas de pose;
- valide se a tabela `face_embeddings` foi populada.

### 9.2 Gráficos ou relatórios sem dados

- verifique os filtros de aluno, disciplina, turma e data;
- confirme se existe sessão em `monitoring_sessions`;
- confirme se existem episódios em `behavior_episode`;
- verifique se o usuário logado tem escopo para enxergar aqueles dados.

### 9.3 Professor não consegue cadastrar ou monitorar

- confirme se o usuário está com perfil `professor`;
- verifique se existe registro correspondente em `teachers`;
- confirme se há vínculos em `teacher_subject_class`;
- sem vínculo de turma e disciplina, o sistema restringe as opções da interface.

### 9.4 Delay no vídeo

- reduza `RELAY_SEND_FPS`;
- use stream RTSP secundário com menor resolução;
- confira carga de CPU/GPU;
- evite múltiplos consumidores simultâneos do mesmo stream.

### 9.5 Sessão cai após atualizar o navegador

- revise a configuração de cookie do ambiente;
- se necessário, habilite fallback por token com `AUTH_ENABLE_QUERY_TOKEN`;
- defina um `AUTH_TOKEN_SECRET` seguro fora do valor padrão.

## 10. Boas práticas operacionais

- após novos cadastros, valide o status dos embeddings antes de iniciar monitoramento;
- mantenha turmas, disciplinas e vínculos professor-turma atualizados no banco;
- use tipo de aula corretamente para melhorar as análises comparativas;
- para apresentação, gere dados simulados antes do evento;
- acompanhe logs de `app`, `relay` e `db` em caso de falha;
- não misture banco local e banco Docker sem revisar `DB_HOST`.
