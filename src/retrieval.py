# Standard-Semantische Suche mit FAISS mit retrieve()
# Hybrid Retrieval (Semantik + Keywords + Boosting) mit retrieve_hybrid()

import numpy as np
import re
from src.embedding_local import embed_texts
from src.faiss_store import load_faiss_index

# spaCy kann fehlen – fallback sicherstellen
try:
    from src.synonym_utils import expand_with_synonyms
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False


def retrieve(query: str, top_k: int = 5, variant: str = "basegame"):
    """
    Klassische semantische Suche basierend auf FAISS.
    """
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    if len(I[0]) == 0:
        return []
    return [metadata[i] for i in I[0]]


def retrieve_hybrid(query: str, top_k: int = 5, variant: str = "basegame", alpha: float = 0.6,
                    expected_keywords=None, debug: bool = False):
    """
    Hybrid Retrieval: kombiniert semantische Ähnlichkeit (FAISS) mit Keyword-Score.
    alpha ∈ [0, 1]: Gewichtung für dense similarity.
    expected_keywords: Liste oder Menge von Schlüsselwörtern (strings), die bei Treffer den Score boosten.
    """
    index, metadata = load_faiss_index(variant)
    query_vec = embed_texts([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k * 3)  # mehr Kandidaten für bessere Mischung

    # Fallback wenn spaCy fehlt
    if HAS_SPACY:
        try:
            query_terms = expand_with_synonyms(query)
        except Exception:
            query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
    else:
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

        # Bonus wenn erwartete Keywords enthalten sind
        if expected_keywords and any(kw.lower() in chunk.lower() for kw in expected_keywords):
            hybrid_score += 0.1

        # Score cap bei 1.0
        hybrid_score = min(hybrid_score, 1.0)

        if debug:
            print(f"Chunk: {chunk[:80]}...")
            print(f"Hybrid Score: {hybrid_score:.3f} (Dense: {dense_score:.3f}, Keyword: {keyword_score:.3f})")

        results.append({
            "chunk": chunk,
            "hybrid_score": hybrid_score,
            "dense_score": dense_score,
            "keyword_score": keyword_score
        })

    # Top-k nach Score sortiert
    return sorted(results, key=lambda r: -r["hybrid_score"])[:top_k]


def retrieve_across_variants(query: str, top_k: int = 3, expected_number: str = None,
                              preferred_variant: str = None, use_hybrid: bool = False):
    """
    Vergleicht den Query über alle Spielvarianten hinweg. Optional: hybrid mode aktivierbar.
    """
    query_vec = embed_texts([query])[0].astype("float32")

    variants = [
        "basegame",
        "seafarers",
        "cities_knights",
        "traders_barbarians",
        "explorers_pirates"
    ]
    results = []

    for variant in variants:
        if use_hybrid:
            variant_results = retrieve_hybrid(query, top_k=1, variant=variant)
            if not variant_results:
                continue
            best = variant_results[0]
            best_chunk = best["chunk"]
            best_score = best["hybrid_score"]
        else:
            index, metadata = load_faiss_index(variant)
            D, I = index.search(np.array([query_vec]), top_k)
            if len(I[0]) == 0:
                continue
            best_chunk = metadata[I[0][0]]
            best_score = 1 - D[0][0]  # L2-Distanz → Ähnlichkeit

        # Kleine Score-Booster für bevorzugte Variante oder erwartete Begriffe im Chunk
        if preferred_variant and variant == preferred_variant:
            best_score += 0.05

        if expected_number and expected_number in best_chunk:
            best_score += 0.05

        results.append({
            "variant": variant,
            "similarity": round(float(best_score), 4),
            "chunk": best_chunk
        })

    return sorted(results, key=lambda x: -x["similarity"])