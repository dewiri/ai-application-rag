import numpy as np
import re
from src.embedding_local import embed_texts
from src.faiss_store import load_faiss_index

def retrieve(query: str, top_k: int = 5, variant: str = "basegame"):
    """
    Klassische semantische Suche basierend auf FAISS.
    """
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    return [metadata[i] for i in I[0]]

def retrieve_hybrid(query: str, top_k: int = 5, variant: str = "basegame", alpha: float = 0.8):
    """
    Hybrid Retrieval: kombiniert semantische Ähnlichkeit (FAISS) mit Keyword-Score.
    
    alpha ∈ [0, 1]: Gewichtung für dense similarity.
    (z. B. 0.8 = 80% semantisch, 20% keyword-basiert)
    """
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k * 2)  # mehr holen für bessere Mischung

    query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))

    results = []
    for dist, idx in zip(D[0], I[0]):
        chunk = metadata[idx]
        chunk_terms = set(re.findall(r"\b\w{3,}\b", chunk.lower()))

        # Keyword-Overlap (Jaccard)
        keyword_score = len(query_terms & chunk_terms) / len(query_terms | chunk_terms) if query_terms else 0.0

        # Dense-Similarity (L2 -> Similarity)
        dense_score = max(0.0, min(1.0, 1 - dist))

        # Hybrid-Gewichtung
        hybrid_score = alpha * dense_score + (1 - alpha) * keyword_score

        results.append({
            "chunk": chunk,
            "hybrid_score": hybrid_score,
            "dense_score": dense_score,
            "keyword_score": keyword_score
        })

    # Top-N nach kombinierten Scores
    return sorted(results, key=lambda r: -r["hybrid_score"])[:top_k]