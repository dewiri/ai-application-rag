# Cross-Variant Similarity-Analyse

import numpy as np
from src.embedding_local import embed_texts
from src.faiss_store import load_faiss_index


def retrieve_across_variants(query: str, top_k: int = 3, expected_number: str = None, preferred_variant: str = None):
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
        index, metadata = load_faiss_index(variant)
        D, I = index.search(np.array([query_vec]), top_k)

        best_chunk = metadata[I[0][0]]
        best_score = 1 - D[0][0]  # L2-Distanz → Ähnlichkeit

        # Optionaler Boost für bevorzugte Variante
        if preferred_variant and variant == preferred_variant:
            best_score += 0.05

        # Optional: Bonus wenn erwartete Zahl im Text vorkommt
        if expected_number and expected_number in best_chunk:
            best_score += 0.05

        results.append({
            "variant": variant,
            "similarity": round(float(best_score), 4),
            "chunk": best_chunk
        })

    return sorted(results, key=lambda x: -x["similarity"])