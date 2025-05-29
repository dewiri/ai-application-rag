# Altes Re-Ranking Modul. Sortiert semantisch die ähnlichsten Texte zur Nutzeranfrage nach dem initialen Retrieval nochmal.

from sentence_transformers import CrossEncoder

# Cross-Encoder für finales Re-Ranking initialisieren
reranker = CrossEncoder("cross-encoder/ms-marco-electra-base")

def rerank(query: str, docs: list[str], top_n: int = 3) -> list[str]:
    """
    Re-Rankt eine Liste von Dokumenten anhand des Queries
    und gibt die besten top_n zurück.
    """
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]