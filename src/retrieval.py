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

def retrieve_hybrid(query: str, top_k: int = 5, variant: str = "basegame", alpha: float = 0.6, expected_keywords=None):
    """
    Hybrid Retrieval: kombiniert semantische Ähnlichkeit (FAISS) mit Keyword-Score.
    
    alpha ∈ [0, 1]: Gewichtung für dense similarity.
    (z. B. 0.6 = 60% semantisch, 40% keyword-basiert)

    expected_keywords: Liste oder Menge von Schlüsselwörtern (strings), die bei Treffer den Score boosten.
    """
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k * 3)  # mehr Kandidaten für bessere Mischung

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

        # Bonus wenn eines der erwarteten Keywords im Chunk vorkommt (Groß-/Kleinschreibung ignorieren)
        if expected_keywords and any(kw.lower() in chunk.lower() for kw in expected_keywords):
            hybrid_score += 0.1 

        # Score cap bei 1.0
        hybrid_score = min(hybrid_score, 1.0)

        results.append({
            "chunk": chunk,
            "hybrid_score": hybrid_score,
            "dense_score": dense_score,
            "keyword_score": keyword_score
        })

    # Sortieren und Top-k zurückgeben
    results = sorted(results, key=lambda r: -r["hybrid_score"])[:top_k]
    return results