import numpy as np
from src.embedding_local import embed_texts
from src.faiss_store import load_faiss_index

def retrieve(query: str, top_k: int = 5, variant: str = "basegame"):
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    return [metadata[i] for i in I[0]]