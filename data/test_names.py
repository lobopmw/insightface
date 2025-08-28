import numpy as np
import pickle

# Caminhos dos arquivos
embeddings_path = "embeddings.npy"
names_path = "names.pkl"

# Carregar embeddings
embeddings = np.load(embeddings_path)

# Carregar nomes
with open(names_path, "rb") as f:
    names = pickle.load(f)

print("==== VERIFICAÇÃO FACEBANK ====")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Total de nomes: {len(names)}")
print("Primeiros nomes:", names[:5])
