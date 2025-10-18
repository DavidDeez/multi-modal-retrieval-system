import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FAISSSearchEngine:
    def __init__(self, dimension, use_gpu=False):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_vectors(self, embeddings, metadata):
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=20, diversity_threshold=0.95):
        query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(query_vector, k * 3)

        selected_indices = []
        selected_embeddings = []

        for idx in indices[0]:
            if len(selected_indices) >= k:
                break

            candidate_emb = self.index.reconstruct(int(idx))

            if not selected_embeddings:
                selected_indices.append(idx)
                selected_embeddings.append(candidate_emb)
                continue

            sims = cosine_similarity([candidate_emb], selected_embeddings)[0]
            if np.max(sims) < diversity_threshold:
                selected_indices.append(idx)
                selected_embeddings.append(candidate_emb)

        results = []
        for i, idx in enumerate(selected_indices):
            results.append({
                'metadata': self.metadata[idx],
                'similarity': float(distances[0][np.where(indices[0] == idx)[0][0]]),
                'rank': i + 1
            })
        return results
