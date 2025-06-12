import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self, texts):
        self.texts = texts
        self.embeddings = embedder.encode(texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query, top_k=3):
        q_embed = embedder.encode([query])
        distances, indices = self.index.search(q_embed, top_k)
        return [self.texts[i] for i in indices[0]]
