import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def identify_face(embedding, known_embeddings, known_names, threshold=0.5):
    if len(known_embeddings) == 0:
        return "Desconhecido"

    sims = cosine_similarity([embedding], known_embeddings)[0]
    best_match_idx = np.argmax(sims)
    best_score = sims[best_match_idx]

    if best_score >= threshold:
        return known_names[best_match_idx]
    else:
        return "Desconhecido"
