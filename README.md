# Monitoramento de Comportamentos em Sala

Sistema de monitoramento em tempo real com **YOLO Pose** + **InsightFace**, streaming **RTSP → Socket (relay)**, e interface em **Streamlit**.

## 🧱 Requisitos

- Python 3.9+  
- GPU opcional (CUDA) para acelerar o InsightFace/YOLO
- Câmera IP com RTSP (ex.: Hikvision/Intelbras)
- VLC/ffplay (opcional, pra testar o RTSP)

### Dependências (pip)
```bash
pip install -r requirements.txt
```
Se não tiver `requirements.txt`, instale:
```bash
pip install streamlit ultralytics opencv-python numpy torch torchvision torchaudio             scikit-learn pillow insightface onnxruntime-gpu onnxruntime             pandas plotly reportlab
```
> Use **onnxruntime-gpu** se tiver CUDA. Senão, deixe só **onnxruntime**.

## 📁 Estrutura recomendada de pastas

```
project/
│
├─ src/
│   ├─ insightface_classroom.py        # app Streamlit (principal)
│   ├─ socket_video_stream.py          # cliente do relay (VideoStream)
│   ├─ server_rtsp_socket.py           # servidor relay RTSP → socket (exemplo)
│   ├─ register_face_multi_images_avg.py
│   ├─ control_database.py             # DB e gráficos
│   ├─ utils_criptografia.py           # salvar_mapeamento (hash)
│   └─ images/
│       ├─ classroom1.jpg
│       ├─ faces.png
│       ├─ cam_IA.png
│       └─ table.png
│
└─ data/
    ├─ mapeamento_alunos.csv           # CSV global (fora da pasta 'alunos')
    ├─ alunos/
    │   └─ <hash_do_aluno>/
    │       ├─ frontal/
    │       ├─ lateral_direita/
    │       ├─ lateral_esquerda/
    │       └─ cabeca_baixa/
    ├─ embeddings.npy                  # gerado pelo script de registro
    └─ names.json / names.pkl          # gerado pelo script de registro
```

> **Importante:** O app já usa `data/` como raiz, com:
> - `data/mapeamento_alunos.csv`
> - `data/alunos/<hash>/...`

## 🚀 Passo a passo (primeira execução)

### 1) Teste rápido do RTSP (opcional)
Confirme que sua URL RTSP funciona:
```bash
vlc "rtsp://usuario:senha@IP_DA_CAMERA:554/Streaming/Channels/101"
# ou:
ffplay -rtsp_transport tcp "rtsp://usuario:senha@IP:554/Streaming/Channels/101"
```

### 2) Inicie o **servidor relay** (RTSP → socket)
No terminal A:
```bash
cd src
python server_rtsp_socket.py   --rtsp "rtsp://usuario:senha@IP_DA_CAMERA:554/Streaming/Channels/101"   --host 0.0.0.0   --port 5555   --send-fps 15   --resize 1280x720
```
- **--send-fps**: taxa de quadros enviada ao cliente  
- **--resize**: opcional (diminui resolução pra reduzir latência/CPU)
- **host/port**: mantenha coerente com o cliente (**127.0.0.1:5555** por padrão)

> Dica: para **menos delay**, use o **substream** da câmera (ex.: *Channels/102*), GOP curto (I-frame mais frequente), bitrate CBR moderado, desative “smart codecs”.

### 3) Abra o **app Streamlit**
No terminal B:
```bash
cd src
streamlit run insightface_classroom.py
```
Na interface:
- Menu → **Cadastro de Alunos**
- Depois → **Monitoramento**
- **Gráficos** e **Tabela** para análise/relatórios

## 👤 Cadastro de alunos (automático por pose)

1. Abra **Cadastro de Alunos**
2. Preencha **Disciplina**, **Nome** e **Matrícula**
3. Ajuste:
   - **Imagens por pose** (padrão 10)
   - **Intervalo entre fotos (seg)** (padrão 0.8s)
   - **Contagem inicial (seg)** (padrão 2s) → “respira, posiciona, valendo!”
4. Clique **Iniciar captura desta pose**  
   O app salva automaticamente as N imagens da pose na pasta:
   ```
   data/alunos/<hash_aluno>/<pose>/
   ```
5. Clique **Próximo** para a pose seguinte até concluir todas.  
6. Ao final, clique **Finalizar cadastro**.

> O arquivo **data/mapeamento_alunos.csv** guarda `nome`, `matricula`, `hash` do aluno.

## 🧠 Gerar/atualizar embeddings (faces conhecidas)

Sempre que cadastrar/alterar alunos, **rode**:
```bash
cd src
python register_face_multi_images_avg.py
```
Esse script varre `data/alunos/`, calcula os embeddings médios por aluno e escreve:
- `data/embeddings.npy`
- `data/names.json` (ou `.pkl`, conforme seu script)

O app carrega isso com `load_insightface_data()`.

## 👁️ Monitoramento

1. Certifique-se de que o **relay** (servidor socket) está **rodando**.
2. No app, vá em **Monitoramento** e clique **Iniciar Monitoramento**.
3. Opções:
   - **Confiança Mínima** (pose)
   - **Usar GPU (CUDA)** se disponível
4. O app reconhece o aluno pelo rosto (se estiver no banco de embeddings) e classifica:
   - **Atento**, **Perguntando**, **Escrevendo**, **Dormindo**, **Agitado**  
   - **Distraído** é aplicado por tempo se a cabeça ficar de lado (regras simples)

As mudanças de comportamento são enviadas para o banco via `insert_count_behavior(...)`.

## 📊 Gráficos e 📋 Tabela

- **Gráficos**: selecione **Aluno**, **Disciplina** e **Data**  
  - Distribuição por comportamento  
  - Contagem por comportamento  
  - Evolução temporal  
  - Botões para **download PNG** e **PDF** com os três gráficos
- **Tabela**: filtragem por data, disciplina e comportamento

## ⚙️ Notas de desempenho / baixa latência

- Prefira **substream** (ex.: *Channels/102*) para reduzir bitrate/resolução.  
- GOP curto (I-frame mais frequente), CBR moderado, **TCP** no RTSP.  
- No **servidor relay**:
  - Ajuste `--send-fps` e `--resize`.  
  - Evite enviar mais FPS do que o necessário.
- No **app**:
  - Mantemos um **fps de render** fixo e um **worker** de IA que sempre processa o frame **mais recente** (evita fila).
  - InsightFace: no GPU, `ctx_id=0`; no CPU, `ctx_id=-1` e `det_size=(640,640)` pra acelerar.
- Evite múltiplas instâncias do app/relay usando a mesma porta.  
- Se notar atraso após “rerun”, confira se o **relay** e a **webcam de cadastro** foram fechados (o app já faz isso ao mudar de página, mas vale checar).

## 🛠️ Solução de problemas

- **Sem vídeo no monitoramento**  
  - Verifique o log do **servidor relay**  
  - IP/porta corretos? (app usa `("127.0.0.1", 5555)`)  
  - Firewall liberado?
- **Muito delay**  
  - Reduza `--resize` ou `--send-fps` no servidor  
  - Use substream da câmera  
  - Verifique que não há outra janela/consumidor do RTSP em paralelo
- **Aluno como “Desconhecido”**  
  - Rode novamente `register_face_multi_images_avg.py` após novos cadastros  
  - Tire mais imagens **frontais** (bem iluminadas)  
  - Verifique o **threshold** de similaridade (padrão ~0.45 na app)
- **Contagens não aparecem nos gráficos/tabela**  
  - Os gráficos filtram por **Aluno**, **Disciplina** e **Data** — confira se coincidem  
  - Confirme se `insert_count_behavior(...)` está sendo chamado (muda de estado)  
  - Cheque a base/arquivo SQLite configurado em `control_database.py`

## 🔒 Observações de privacidade

- O app guarda um **hash** do aluno (nome + matrícula) para nomear as pastas.  
- As fotos ficam em `data/alunos/<hash>/...`.  
- Proteja a pasta `data/` e o acesso à máquina/servidor.

## 🧪 Comandos úteis (resumo)

```bash
# 1) Relay (servidor)
python src/server_rtsp_socket.py --rtsp "rtsp://user:pass@CAM_IP:554/Streaming/Channels/101"   --host 0.0.0.0 --port 5555 --send-fps 15 --resize 1280x720

# 2) App
streamlit run src/insightface_classroom.py

# 3) Atualizar embeddings após cadastro
python src/register_face_multi_images_avg.py
```

## ✍️ Dicas finais

- Para evitar “rerun” com streams abertos, sempre **pare o monitoramento** antes de mexer em opções/voltar pro cadastro.
- Se trocar **host/porta** do relay, atualize no `VideoStream(("HOST", PORTA))`.
- Se precisar mudar caminhos, ajuste no início do app:
  - `DATA_DIR`, `DATABASE_PATH`, `MAPPING_CSV`.
