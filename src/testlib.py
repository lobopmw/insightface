# import os
# import cv2
# from insightface.app import FaceAnalysis

# # Inicializa o modelo InsightFace
# app = FaceAnalysis(providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=-1, det_size=(640, 640))

# # 🔁 Defina as variáveis corretamente
# hash_pasta = "d4c72a593d936e48b6d8ddcbb4c34bc7bfcd88dc0a066c90fbf800fe55db4ef3"  # Ex: "fa35d3cfedb91a..."
# nome_imagem = "cabeca_baixa_20250710_005828043087.jpg"  # Nome da imagem dentro da subpasta "frontal"

# # 📂 Monta o caminho absoluto da imagem de forma segura
# img_path = os.path.join("data", "alunos", hash_pasta, "cabeca_baixa", nome_imagem)

# # 🖼️ Carrega e processa a imagem
# img = cv2.imread(img_path)

# if img is None:
#     print(f"Erro ao carregar imagem em: {img_path}")
# else:
#     faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     print(f"✅ Rostos detectados: {len(faces)}")


import pandas as pd
import os

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mapeamento_alunos.csv'))
print("Usando:", csv_path)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
else:
    print("❌ Arquivo CSV não encontrado.")
